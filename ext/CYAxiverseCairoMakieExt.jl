"""
    CYAxiverseCairoMakieExt

Optional CairoMakie implementation of the CYAxiverse plotting pipeline. The
extension is loaded when users explicitly load both `CairoMakie` and
`ColorSchemes`.
"""
module CYAxiverseCairoMakieExt

using CYAxiverse
using CairoMakie
using ColorSchemes
using Dates

import CYAxiverse.plotting: PlotResult, PlotStyle, Curve, Band, ReferenceLine,
    curve, boxplot, scatterplot, exclusionplot, functionplot, minima_plot, trajectoryplot,
    save_plot, styled_axis, vacua_db_jlm, vacua_db_jlm_single, total_geometries,
    vacua_db_jlm_box
using CYAxiverse.filestructure: plots_dir, count_geometries, paths_cy
using CYAxiverse.generate: jlm_vacua_db

const _EMPTY_KWARGS = NamedTuple()
const _FONT_AXIS_ATTRIBUTES = (
    :titlefont, :xlabelfont, :ylabelfont, :xticklabelfont, :yticklabelfont,
)
const _COLOR_AXIS_ATTRIBUTES = (
    :backgroundcolor, :xgridcolor, :ygridcolor, :xlabelcolor, :ylabelcolor,
    :titlecolor, :xticklabelcolor, :yticklabelcolor, :xtickcolor, :ytickcolor,
)

function _kw(kwargs::NamedTuple, name::Symbol, default)
    return haskey(kwargs, name) ? getproperty(kwargs, name) : default
end

function _axis_defaults(style::PlotStyle; xscale = identity, yscale = identity)
    return (
        backgroundcolor = style.background,
        xgridcolor = style.gridcolor,
        ygridcolor = style.gridcolor,
        xgridvisible = true,
        ygridvisible = true,
        xminorticksvisible = true,
        yminorticksvisible = true,
        xminorgridvisible = true,
        yminorgridvisible = true,
        spinewidth = 1.2,
        xlabelcolor = style.foreground,
        ylabelcolor = style.foreground,
        titlecolor = style.foreground,
        xticklabelcolor = style.foreground,
        yticklabelcolor = style.foreground,
        xtickcolor = style.foreground,
        ytickcolor = style.foreground,
        xlabelfont = style.font,
        ylabelfont = style.font,
        topspinevisible = true,
        rightspinevisible = true,
        xticklabelfont = style.font,
        yticklabelfont = style.font,
        titlefont = style.font,
        xscale = xscale,
        yscale = yscale,
    )
end

function _set_axis_attribute!(axis, name::Symbol, value)
    if name in _FONT_AXIS_ATTRIBUTES && value isa AbstractString
        # Axis stores font roles as Symbols after construction. The constructor
        # accepts a font name, but Makie does not convert a String on mutation.
        return nothing
    elseif name in _COLOR_AXIS_ATTRIBUTES
        setproperty!(axis, name, Makie.to_color(value))
    else
        setproperty!(axis, name, value)
    end
    return nothing
end

"""
    styled_axis(position; style = PlotStyle(), kwargs...)

Create a CairoMakie `Axis` with the shared CYAxiverse style. Use this helper
when composing several panels in a manually-created `Figure`; renderer
functions can then safely receive the resulting axis through `axis = ...`.
"""
function styled_axis(position; style::PlotStyle = PlotStyle(),
    axis_kwargs::NamedTuple = _EMPTY_KWARGS, xscale = identity, yscale = identity)

    attributes = merge(_axis_defaults(style; xscale = xscale, yscale = yscale), axis_kwargs)
    return Axis(position; attributes...)
end

function _prepare_axis(; style::PlotStyle, axis, axis_kwargs::NamedTuple,
    xlabel = nothing, ylabel = nothing, title = nothing,
    xscale = identity, yscale = identity)

    if isnothing(axis)
        figure = Figure(
            size = style.resolution,
            fontsize = style.fontsize,
            figure_padding = style.figure_padding,
            backgroundcolor = style.background,
        )
        defaults = _axis_defaults(style; xscale = xscale, yscale = yscale)
        axis_attributes = merge(defaults, axis_kwargs)
        axis_object = styled_axis(figure[1, 1]; style = style,
            axis_kwargs = merge(axis_attributes, (;)), xscale = xscale, yscale = yscale)
        return figure, axis_object
    end

    for (name, value) in pairs(_axis_defaults(style))
        _set_axis_attribute!(axis, name, value)
    end
    for (name, value) in pairs(axis_kwargs)
        _set_axis_attribute!(axis, name, value)
    end
    if xscale !== identity
        axis.xscale = xscale
    end
    if yscale !== identity
        axis.yscale = yscale
    end
    return nothing, axis
end

function _set_decorations!(axis; xlabel = nothing, ylabel = nothing, title = nothing)
    !isnothing(xlabel) && (axis.xlabel = xlabel)
    !isnothing(ylabel) && (axis.ylabel = ylabel)
    !isnothing(title) && (axis.title = title)
    return axis
end

function _maybe_legend!(axis, show_legend::Bool, legend_kwargs::NamedTuple)
    show_legend && axislegend(axis; legend_kwargs...)
    return nothing
end

function _primary_plot(plots::Tuple)
    return length(plots) == 1 ? first(plots) : plots
end

function _curve_plot_kwargs(style::PlotStyle, plot_kwargs::NamedTuple, index::Int)
    color = style.accent_colors[mod1(index, length(style.accent_colors))]
    return merge((; color = color, linewidth = 2.5), plot_kwargs)
end

"""
    boxplot(groups; kwargs...)

Render one Tukey box-and-whisker distribution per entry in `groups`. Use
`positions` for numeric category locations and `labels` for their displayed
names. Pass `orientation = :horizontal` for the paper's horizontal layout.

The renderer returns a `PlotResult`. Provide an existing Makie `axis` to add
the boxes to a multi-panel figure.
"""
function boxplot(groups::AbstractVector{<:AbstractVector}; style::PlotStyle = PlotStyle(),
    axis = nothing, positions = nothing, labels = nothing, orientation::Symbol = :vertical,
    xlabel = nothing, ylabel = nothing, title = nothing, show_legend::Bool = false,
    legend_kwargs::NamedTuple = _EMPTY_KWARGS, axis_kwargs::NamedTuple = _EMPTY_KWARGS,
    plot_kwargs::NamedTuple = _EMPTY_KWARGS)

    isempty(groups) && throw(ArgumentError("groups must not be empty"))
    orientation in (:vertical, :horizontal) ||
        throw(ArgumentError("orientation must be :vertical or :horizontal"))
    category_positions = isnothing(positions) ? collect(1:length(groups)) : collect(positions)
    length(category_positions) == length(groups) ||
        throw(DimensionMismatch("positions and groups must have equal length"))
    category_labels = isnothing(labels) ? string.(category_positions) : collect(labels)
    length(category_labels) == length(groups) ||
        throw(DimensionMismatch("labels and groups must have equal length"))

    figure, axis_object = _prepare_axis(
        style = style, axis = axis, axis_kwargs = axis_kwargs,
        xlabel = xlabel, ylabel = ylabel, title = title,
    )
    _set_decorations!(axis_object; xlabel = xlabel, ylabel = ylabel, title = title)

    plots = ()
    for (index, values) in enumerate(groups)
        isempty(values) && throw(ArgumentError("boxplot groups must not be empty"))
        x = fill(category_positions[index], length(values))
        attributes = merge((
            color = style.accent_colors[mod1(index, length(style.accent_colors))],
            width = 0.85,
            gap = 0,
            whiskerwidth = 0.75,
            marker = :xcross,
            markersize = 7,
            orientation = orientation,
        ), plot_kwargs)
        plot = CairoMakie.boxplot!(axis_object, x, values; attributes...)
        plots = (plots..., plot)
    end

    if orientation === :vertical
        axis_object.xticks = (category_positions, category_labels)
    else
        axis_object.yticks = (category_positions, category_labels)
    end
    _maybe_legend!(axis_object, show_legend, legend_kwargs)
    return PlotResult(figure, axis_object, _primary_plot(plots))
end

"""
    scatterplot(x, y; kwargs...)

Render a publication-style scatter plot. Supply a numeric `color` vector and
`colorbar = true` to add a colorbar when the function creates the figure.
"""
function scatterplot(x::AbstractVector, y::AbstractVector; style::PlotStyle = PlotStyle(),
    axis = nothing, xlabel = nothing, ylabel = nothing, title = nothing,
    label = "", color = nothing, colormap = style.palette, colorbar::Bool = false,
    colorbar_label = nothing, markersize = 10, show_legend::Bool = false,
    legend_kwargs::NamedTuple = _EMPTY_KWARGS, axis_kwargs::NamedTuple = _EMPTY_KWARGS,
    plot_kwargs::NamedTuple = _EMPTY_KWARGS)

    length(x) == length(y) || throw(DimensionMismatch("x and y must have equal length"))
    figure, axis_object = _prepare_axis(
        style = style, axis = axis, axis_kwargs = axis_kwargs,
        xlabel = xlabel, ylabel = ylabel, title = title,
    )
    _set_decorations!(axis_object; xlabel = xlabel, ylabel = ylabel, title = title)
    point_color = isnothing(color) ? style.accent_colors[1] : color
    attributes = merge((;
        color = point_color,
        colormap = colormap,
        markersize = markersize,
        marker = :circle,
    ), plot_kwargs)
    !isempty(label) && (attributes = merge(attributes, (; label = label)))
    plot = CairoMakie.scatter!(axis_object, x, y; attributes...)

    if colorbar && !isnothing(figure) && point_color isa AbstractVector{<:Real}
        Colorbar(figure[1, 2], plot; label = colorbar_label)
    end
    _maybe_legend!(axis_object, show_legend, legend_kwargs)
    return PlotResult(figure, axis_object, plot)
end

function _band_parts(item::Band)
    return item.x, item.lower, item.upper, item.label, item.color, item.alpha
end

function _band_parts(item::NamedTuple)
    required = (:x, :lower, :upper)
    all(name -> haskey(item, name), required) ||
        throw(ArgumentError("band named tuples require x, lower, and upper"))
    return item.x, item.lower, item.upper, _kw(item, :label, ""),
        _kw(item, :color, "#D1495B"), Float64(_kw(item, :alpha, 0.22))
end

function _reference_parts(item::ReferenceLine)
    return item.value, item.orientation, item.label, item.color, item.linestyle, item.linewidth
end

function _reference_parts(item::NamedTuple)
    haskey(item, :value) || throw(ArgumentError("reference line requires value"))
    orientation = _kw(item, :orientation, :vertical)
    orientation in (:vertical, :horizontal) ||
        throw(ArgumentError("orientation must be :vertical or :horizontal"))
    return item.value, orientation, _kw(item, :label, ""),
        _kw(item, :color, "#222222"), _kw(item, :linestyle, :dash),
        Float64(_kw(item, :linewidth, 1.5))
end

function _draw_reference_line!(axis, item)
    value, orientation, label, color, linestyle, linewidth = _reference_parts(item)
    attributes = (; color = color, linestyle = linestyle, linewidth = linewidth)
    if orientation === :vertical
        return isempty(label) ? vlines!(axis, value; attributes...) :
            vlines!(axis, value; attributes..., label = label)
    end
    return isempty(label) ? hlines!(axis, value; attributes...) :
        hlines!(axis, value; attributes..., label = label)
end

"""
    exclusionplot(x, y; bands = (), reference_lines = (), kwargs...)

Render an exclusion curve with optional `Band` confidence regions and dashed
`ReferenceLine`s. The same function supports a one-dimensional excluded
fraction versus Hodge number and two-dimensional mass/coupling exclusion
curves by choosing suitable axis scales and labels.
"""
function exclusionplot(x::AbstractVector, y::AbstractVector; style::PlotStyle = PlotStyle(),
    axis = nothing, xlabel = nothing, ylabel = nothing, title = nothing,
    label = "", color = style.accent_colors[2], linewidth = 2.8,
    bands = (), reference_lines = (), show_legend::Bool = true,
    legend_kwargs::NamedTuple = _EMPTY_KWARGS, axis_kwargs::NamedTuple = _EMPTY_KWARGS,
    plot_kwargs::NamedTuple = _EMPTY_KWARGS)

    length(x) == length(y) || throw(DimensionMismatch("x and y must have equal length"))
    figure, axis_object = _prepare_axis(
        style = style, axis = axis, axis_kwargs = axis_kwargs,
        xlabel = xlabel, ylabel = ylabel, title = title,
    )
    _set_decorations!(axis_object; xlabel = xlabel, ylabel = ylabel, title = title)

    plots = ()
    for item in reference_lines
        plots = (plots..., _draw_reference_line!(axis_object, item))
    end
    for item in bands
        band_x, lower, upper, band_label, band_color, alpha = _band_parts(item)
        band_attributes = (; color = (band_color, alpha), strokecolor = band_color)
        band_plot = isempty(band_label) ?
            CairoMakie.band!(axis_object, band_x, lower, upper; band_attributes...) :
            CairoMakie.band!(axis_object, band_x, lower, upper;
                band_attributes..., label = band_label)
        plots = (plots..., band_plot)
    end
    line_attributes = merge((; color = color, linewidth = linewidth), plot_kwargs)
    !isempty(label) && (line_attributes = merge(line_attributes, (; label = label)))
    line_plot = CairoMakie.lines!(axis_object, x, y; line_attributes...)
    plots = (plots..., line_plot)
    _maybe_legend!(axis_object, show_legend, legend_kwargs)
    return PlotResult(figure, axis_object, _primary_plot(plots))
end

function _functionplot(curves::Tuple; style::PlotStyle, axis,
    xlabel, ylabel, title, show_legend, legend_kwargs, axis_kwargs,
    plot_kwargs, xscale, yscale)

    isempty(curves) && throw(ArgumentError("functionplot requires at least one curve"))
    figure, axis_object = _prepare_axis(
        style = style, axis = axis, axis_kwargs = axis_kwargs,
        xlabel = xlabel, ylabel = ylabel, title = title,
        xscale = xscale, yscale = yscale,
    )
    _set_decorations!(axis_object; xlabel = xlabel, ylabel = ylabel, title = title)

    plots = ()
    for (index, sampled_curve) in enumerate(curves)
        attributes = _curve_plot_kwargs(style, plot_kwargs, index)
        if !isempty(sampled_curve.label)
            attributes = merge(attributes, (; label = sampled_curve.label))
        end
        plot = CairoMakie.lines!(axis_object, sampled_curve.x, sampled_curve.y; attributes...)
        plots = (plots..., plot)
    end
    _maybe_legend!(axis_object, show_legend, legend_kwargs)
    return PlotResult(figure, axis_object, _primary_plot(plots))
end

"""
    functionplot(x, y; kwargs...)
    functionplot(f, x; kwargs...)
    functionplot(curves; kwargs...)

Render one or more sampled functions. The callable form samples `f` on `x`;
the `Curve` form supports several functions with independent sampling grids.
All forms accept `xscale`, `yscale`, and an existing `axis` for composition.
"""
function functionplot(x::AbstractVector, y::AbstractVector; style::PlotStyle = PlotStyle(),
    axis = nothing, label = "", xlabel = nothing, ylabel = nothing, title = nothing,
    xscale = identity, yscale = identity, show_legend::Bool = false,
    legend_kwargs::NamedTuple = _EMPTY_KWARGS, axis_kwargs::NamedTuple = _EMPTY_KWARGS,
    plot_kwargs::NamedTuple = _EMPTY_KWARGS)

    return _functionplot((curve(x, y; label = label),); style = style, axis = axis,
        xlabel = xlabel, ylabel = ylabel, title = title, show_legend = show_legend,
        legend_kwargs = legend_kwargs, axis_kwargs = axis_kwargs,
        plot_kwargs = plot_kwargs, xscale = xscale, yscale = yscale)
end

function functionplot(f::Function, x::AbstractVector; kwargs...)
    return functionplot(x, f.(x); kwargs...)
end

function functionplot(curves::AbstractVector{<:Curve}; kwargs...)
    return functionplot(Tuple(curves); kwargs...)
end

function functionplot(curves::Tuple; style::PlotStyle = PlotStyle(), axis = nothing,
    xlabel = nothing, ylabel = nothing, title = nothing, xscale = identity, yscale = identity,
    show_legend::Bool = true, legend_kwargs::NamedTuple = _EMPTY_KWARGS,
    axis_kwargs::NamedTuple = _EMPTY_KWARGS, plot_kwargs::NamedTuple = _EMPTY_KWARGS)

    all(item -> item isa Curve, curves) ||
        throw(ArgumentError("tuple function plots must contain Curve objects"))
    return _functionplot(curves; style = style, axis = axis, xlabel = xlabel,
        ylabel = ylabel, title = title, show_legend = show_legend,
        legend_kwargs = legend_kwargs, axis_kwargs = axis_kwargs,
        plot_kwargs = plot_kwargs, xscale = xscale, yscale = yscale)
end

function _point_data(x::AbstractVector, y::AbstractVector, points)
    isnothing(points) && return nothing
    points == () && return nothing
    if points isa AbstractVector{<:Integer}
        return x[points], y[points]
    elseif points isa Tuple && length(points) == 2
        return points[1], points[2]
    elseif points isa NamedTuple && haskey(points, :x) && haskey(points, :y)
        return points.x, points.y
    end
    throw(ArgumentError("points must be indices or an (x, y) pair"))
end

"""
    minima_plot(x, y; minima = (), critical = (), kwargs...)

Plot a potential or other function and mark minimum and critical-point
locations. `minima` and `critical` may be index vectors or `(x, y)` pairs,
which keeps the presentation independent of how a solver stores its results.
"""
function minima_plot(x::AbstractVector, y::AbstractVector; style::PlotStyle = PlotStyle(),
    axis = nothing, xlabel = nothing, ylabel = nothing, title = nothing,
    minima = (), critical = (), minima_label = "Minimum", critical_label = "Critical",
    show_legend::Bool = true, legend_kwargs::NamedTuple = _EMPTY_KWARGS,
    axis_kwargs::NamedTuple = _EMPTY_KWARGS, plot_kwargs::NamedTuple = _EMPTY_KWARGS)

    result = functionplot(x, y; style = style, axis = axis, xlabel = xlabel,
        ylabel = ylabel, title = title, show_legend = false,
        axis_kwargs = axis_kwargs, plot_kwargs = plot_kwargs)
    plotted = (; curve = result.plot, minimum = nothing, critical = nothing)
    minimum_points = _point_data(x, y, minima)
    if !isnothing(minimum_points)
        minimum_plot = scatter!(result.axis, minimum_points[1], minimum_points[2];
            color = style.accent_colors[1], marker = :circle, markersize = 11,
            label = minima_label)
        plotted = merge(plotted, (; minimum = minimum_plot))
    end
    critical_points = _point_data(x, y, critical)
    if !isnothing(critical_points)
        critical_plot = scatter!(result.axis, critical_points[1], critical_points[2];
            color = style.accent_colors[4], marker = :xcross, markersize = 12,
            label = critical_label)
        plotted = merge(plotted, (; critical = critical_plot))
    end
    _maybe_legend!(result.axis, show_legend, legend_kwargs)
    return PlotResult(result.figure, result.axis, plotted)
end

"""
    trajectoryplot(time, coordinates; kwargs...)

Plot one or more columns of a trajectory matrix against `time`. A vector is
treated as a single coordinate. Use `event_indices` to mark solver events or
milestones on every coordinate.
"""
function trajectoryplot(time::AbstractVector, coordinates::AbstractMatrix;
    style::PlotStyle = PlotStyle(), axis = nothing, labels = (), xlabel = nothing,
    ylabel = nothing, title = nothing, event_indices = (), event_color = nothing,
    show_legend::Bool = true, legend_kwargs::NamedTuple = _EMPTY_KWARGS,
    axis_kwargs::NamedTuple = _EMPTY_KWARGS, plot_kwargs::NamedTuple = _EMPTY_KWARGS)

    size(coordinates, 1) == length(time) ||
        throw(DimensionMismatch("time and trajectory rows must have equal length"))
    label_values = isempty(labels) ? fill("", size(coordinates, 2)) : collect(labels)
    length(label_values) == size(coordinates, 2) ||
        throw(DimensionMismatch("labels and trajectory columns must have equal length"))
    curves = Tuple(
        curve(time, view(coordinates, :, column); label = label_values[column])
        for column in axes(coordinates, 2)
    )
    result = functionplot(curves; style = style, axis = axis, xlabel = xlabel,
        ylabel = ylabel, title = title, show_legend = false,
        legend_kwargs = legend_kwargs, axis_kwargs = axis_kwargs,
        plot_kwargs = plot_kwargs)

    event_plot = nothing
    if event_indices != () && !isempty(event_indices)
        event_color_value = isnothing(event_color) ? style.accent_colors[1] : event_color
        event_x = repeat(time[event_indices], size(coordinates, 2))
        event_y = vec(coordinates[event_indices, :])
        event_plot = scatter!(result.axis, event_x, event_y; color = event_color_value,
            marker = :circle, markersize = 8)
    end
    _maybe_legend!(result.axis, show_legend, legend_kwargs)
    return PlotResult(result.figure, result.axis, (; curves = result.plot, events = event_plot))
end

function trajectoryplot(time::AbstractVector, coordinate::AbstractVector; kwargs...)
    return functionplot(time, coordinate; kwargs...)
end

"""
    save_plot(path, figure_or_result; kwargs...)

Save a CairoMakie figure or a `PlotResult`. Return the output path so save
steps compose naturally in scripts.
"""
function save_plot(path::AbstractString, result::PlotResult; kwargs...)
    isnothing(result.figure) &&
        throw(ArgumentError("save the owning Figure when PlotResult was drawn on an existing axis"))
    CairoMakie.save(path, result.figure; kwargs...)
    return path
end

function save_plot(path::AbstractString, figure; kwargs...)
    CairoMakie.save(path, figure; kwargs...)
    return path
end

save_plot(result::PlotResult, path::AbstractString; kwargs...) = save_plot(path, result; kwargs...)

"""Render the vacuum database summary using the historical CYAxiverse layout."""
function vacua_db_jlm(vac_data::NamedTuple)
    figure = Figure()
    kwargs = (; xticklabelfont = "STIX", yticklabelfont = "STIX",
        xminorticksvisible = true, xminorgridvisible = true,
        yminorticksvisible = true, yminorgridvisible = true)
    ax1 = Axis(figure[2, 1]; xticks = [4, 50, 100, 200, 300, 400, 491],
        xminorticks = IntervalsBetween(5), yscale = Makie.pseudolog10, kwargs...)
    ax2 = Axis(figure[2, 2]; xticks = [4, 50, 100, 200, 300, 400, 491],
        xminorticks = IntervalsBetween(5), yscale = Makie.pseudolog10,
        yticks = [1, 10, 50, 100, 500, 1000, 2000], kwargs...)
    ax3 = Axis(figure[1, 1:3]; xticks = [4, 50, 100, 200, 300, 400, 491],
        yscale = Makie.pseudolog10, kwargs...)
    square_vac = hcat([vcat(item[1], item[4]) for item in vac_data.square]...)
    n_dim_vac = hcat([vcat(item[1], item[4], item[end]) for item in vac_data.n_dim]...)
    n_dim_vac = n_dim_vac[:, sortperm(n_dim_vac[end, :])]
    one_dim_vac = hcat([vcat(item[1], item[4]) for item in vac_data.one_dim]...)
    square_mask = square_vac[2, :] .!= 0
    one_dim_mask = one_dim_vac[2, :] .!= 0
    n_dim_mask = n_dim_vac[2, :] .!= 0
    sc_square = scatter!(ax3, square_vac[1, square_mask], square_vac[2, square_mask],
        marker = :circle, color = :green, markersize = 10, label = "0 extra rows")
    sc_onedim = scatter!(ax1, one_dim_vac[1, one_dim_mask], one_dim_vac[2, one_dim_mask],
        color = :cyan, marker = :utriangle, markersize = 10, label = "1 extra row")
    sc_ndim = scatter!(ax2, n_dim_vac[1, n_dim_mask], n_dim_vac[2, n_dim_mask],
        color = n_dim_vac[3, n_dim_mask], colormap = :thermal, marker = :rect,
        markersize = 10, label = "N extra rows")
    axislegend(ax3, [sc_square, sc_onedim, sc_ndim], [L"$0$", L"$1$", L"$N$"],
        "Number of Extra Rows", orientation = :horizontal, titlefont = "STIX Bold")
    Colorbar(figure[2, 3], limits = (minimum(n_dim_vac[3, n_dim_mask]),
        maximum(n_dim_vac[3, n_dim_mask])), colormap = :thermal, label = L"$N$",
        ticklabelfont = "STIX")
    Label(figure[1:end, 0], L"$N_\mathrm{vacua}$", rotation = π / 2)
    Label(figure[end + 1, 1:end], L"$h^{1,1}$")
    return figure
end

function vacua_db_jlm_single(vac_data::AbstractMatrix)
    size_inches = (60, 45)
    size_pt = 72 .* size_inches
    cmap = :bamako
    figure = Figure(resolution = size_pt, fontsize = 96,
        figure_padding = (5, 10, 5, 50))
    kwargs = (; xticklabelfont = "STIX", yticklabelfont = "STIX",
        xminorticksvisible = true, xminorgridvisible = true,
        yminorticksvisible = true, yminorgridvisible = true)
    axis = Axis(figure[1, 1]; xticks = [1, 50, 100, 200, 300, 400, 491],
        yticks = [1, 10, 20, 30, 40, 54], xminorticks = IntervalsBetween(5),
        xlabel = L"$h^{1,1}$", ylabel = L"$N_\mathrm{vacua}$", kwargs...)
    sorted_data = sortslices(vac_data, dims = 2, by = x -> x[4])
    all_vac = NTuple{3, Int}[]
    for h11 in sort(collect(Set(sorted_data[1, :])))
        for vac in sort(collect(Set(sorted_data[4, :])))
            count = size(sorted_data[:, sorted_data[1, :] .== h11 .&&
                sorted_data[4, :] .== vac], 2)
            push!(all_vac, (Int(h11), Int(vac), count))
        end
    end
    xlims!(axis, (-3, 494))
    all_vac_matrix = hcat([[item...] for item in all_vac]...)
    mask = all_vac_matrix[3, :] .!= 0
    scatter!(axis, all_vac_matrix[1, mask], all_vac_matrix[2, mask],
        colormap = cmap, color = all_vac_matrix[3, mask], marker = :rect,
        markersize = 25)
    Colorbar(figure[1, 2], colormap = cmap,
        limits = (minimum(all_vac_matrix[3, mask]), maximum(all_vac_matrix[3, mask])),
        labelpadding = 40, label = L"\text{No. of geometries}",
        ticks = [1, 200, 400, 600, 800, 1000], ticklabelfont = "STIX", size = 40)
    save(joinpath(plots_dir(), "N_vac_KS_scatter.pdf"), figure, pt_per_unit = 1)
    return figure
end

function vacua_db_jlm(n = size(paths_cy()[2], 2); one_axis = false)
    vac_data = jlm_vacua_db(; n = n)
    vac_square = hcat([vcat(item[1:4]...) for item in vac_data.square]...)
    vac_1d = hcat([vcat(item[1:4]...) for item in vac_data.one_dim]...)
    vac_nd = hcat([vcat(item[1:4]..., item[end]...) for item in vac_data.n_dim]...)
    all_vacua = hcat(vac_square, vac_1d, vac_nd[1:4, :])
    println(size(all_vacua))
    return one_axis ? vacua_db_jlm_single(all_vacua) : vacua_db_jlm(vac_data)
end

function total_geometries(n = nothing)
    total_geoms = count_geometries(n)
    figure = Figure()
    CairoMakie.barplot(figure[1, 1], hcat(total_geoms...)[1, :], hcat(total_geoms...)[2, :])
    save(joinpath(plots_dir(), "total_geometries.pdf"), figure, pt_per_unit = 1)
    return figure
end

"""Render the historical Hodge-number grouped vacuum boxplot."""
function vacua_db_jlm_box(square::AbstractMatrix, one_dim::AbstractMatrix,
    n_dim::AbstractMatrix; display = false, orientation = :horizontal)

    vacua_full = sortslices(hcat(square, one_dim, n_dim[1:4, :]), dims = 2, by = x -> x[2])
    h11list = sort(collect(Set(vacua_full[1, :])))
    max_h11 = min(400, maximum(h11list))
    orientation in (:horizontal, :vertical) ||
        throw(ArgumentError("orientation must be :horizontal or :vertical"))
    size_inches = orientation === :horizontal ? (36, 48) : (48, 36)
    figure = Figure(resolution = 72 .* size_inches, fontsize = 56)
    colors = resample_cmap(:twilight, maximum(h11list))
    groups = [
        filter(h -> h <= floor(max_h11 / 4), h11list),
        filter(h -> floor(max_h11 / 4) < h <= floor(max_h11 / 2), h11list),
        filter(h -> floor(max_h11 / 2) < h <= floor(3 * max_h11 / 4), h11list),
        filter(h -> floor(3 * max_h11 / 4) < h, h11list),
    ]
    axes = orientation === :horizontal ?
        [Axis(figure[1, i]) for i in 1:4] :
        [Axis(figure[i, 1]) for i in 4:-1:1]
    for (axis, h11_group) in zip(axes, groups)
        for h11 in h11_group
            values = vacua_full[end, vacua_full[1, :] .== h11]
            boxplot!(axis, fill(h11, length(values)), values, orientation = orientation,
                marker = :xcross, markersize = 10, whiskerwidth = 0.75, width = 0.9,
                gap = 0, color = colors[h11])
        end
        if orientation === :horizontal
            hideydecorations!(axis)
        end
    end
    if orientation === :horizontal
        Colorbar(figure[1:end, 0], limits = (minimum(h11list), maximum(h11list)),
            colormap = :twilight, ticklabelfont = "STIX",
            ticks = [4, collect(50:50:400)..., 491], nsteps = length(h11list),
            label = L"$h^{1,1}$", flipaxis = false)
        Label(figure[end + 1, :], L"$\mathcal{N}_{\!\!\mathrm{vac}}$")
        for (axis, h11_group) in zip(axes, groups)
            !isempty(h11_group) && ylims!(axis, minimum(h11_group) - 3, maximum(h11_group) + 3)
        end
    else
        Label(figure[:, 0], L"$\mathcal{N}_{\!\!\mathrm{vac}}$", rotation = π / 2)
    end
    return display ? figure : save(joinpath(plots_dir(), string(now(), "-N_vac_KS_box.pdf")),
        figure, pt_per_unit = 1)
end

function vacua_db_jlm_box(vacua_db::NamedTuple; display = false)
    square = hcat(vacua_db.square...)
    one_dim = hcat([item[1:4] for item in vacua_db.one_dim]...)
    n_dim = hcat([[item[1:4]..., item[end]] for item in vacua_db.n_dim if item[4] != 0]...)
    return vacua_db_jlm_box(square, one_dim, n_dim; display = display)
end

end
