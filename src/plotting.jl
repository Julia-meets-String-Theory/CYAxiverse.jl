"""
    plotting

Core plotting data types and the plotting API for CYAxiverse. Rendering is
provided by the optional `CairoMakie` and `ColorSchemes` extension.

Load the renderer with:

```julia
using CYAxiverse
using CairoMakie
using ColorSchemes
const plotting = CYAxiverse.plotting
```
"""
module plotting

export PlotStyle, PlotResult, Curve, Band, ReferenceLine
export paper_style, curve, band, reference_line
export styled_axis
export boxplot, scatterplot, exclusionplot, functionplot, minima_plot
export trajectoryplot, save_plot
export vacua_db_jlm, vacua_db_jlm_single, total_geometries, vacua_db_jlm_box

"""
    PlotStyle(; kwargs...)

Store the shared visual language for publication plots. The defaults follow
the CYAxiverse paper figures: serif math text, a pale lavender-gray plotting
surface, white gridlines, dark spines, and a restrained high-contrast palette.

Use `paper_style(; kwargs...)` for a named constructor that documents the
intent of the default style. Renderer-specific keyword arguments remain
available through each plotting function's `axis_kwargs` and `plot_kwargs`.
"""
struct PlotStyle{C<:Tuple}
    background::String
    foreground::String
    gridcolor::String
    palette::Symbol
    accent_colors::C
    font::String
    fontsize::Float64
    resolution::Tuple{Int, Int}
    figure_padding::NTuple{4, Int}
end

function PlotStyle(; background::AbstractString = "#E8E8F0",
    foreground::AbstractString = "#222222",
    gridcolor::AbstractString = "#FFFFFF",
    palette::Symbol = :viridis,
    accent_colors = ("#F2B701", "#D1495B", "#1F9E89", "#5B3A9B", "#2C7FB8"),
    font::AbstractString = "STIX",
    fontsize::Real = 20,
    resolution = (900, 650),
    figure_padding = (12, 16, 12, 16))

    length(resolution) == 2 || throw(ArgumentError("resolution must have two entries"))
    length(figure_padding) == 4 ||
        throw(ArgumentError("figure_padding must have four entries"))
    colors = Tuple(string.(accent_colors))
    size = (Int(resolution[1]), Int(resolution[2]))
    padding = ntuple(i -> Int(figure_padding[i]), 4)
    return PlotStyle(
        String(background), String(foreground), String(gridcolor), palette,
        colors, String(font), Float64(fontsize), size, padding,
    )
end

"""
    paper_style(; kwargs...)

Return the default paper-style `PlotStyle`, optionally overriding any style
field. Pass the result to all plotting functions in a figure set to keep
typography, colors, and geometry consistent.
"""
paper_style(; kwargs...) = PlotStyle(; kwargs...)

"""
    Curve(x, y; label = "")

Store one sampled function curve for `functionplot`. `x` and `y` must have the
same length; the sampled arrays are retained without narrowing their numeric
type.
"""
struct Curve{TX<:AbstractVector, TY<:AbstractVector}
    x::TX
    y::TY
    label::String
end

function curve(x::AbstractVector, y::AbstractVector; label = "")
    length(x) == length(y) || throw(DimensionMismatch("x and y must have equal length"))
    return Curve(x, y, String(label))
end

"""
    Band(x, lower, upper; label = "", color = "#D1495B", alpha = 0.22)

Store a confidence or exclusion band for `exclusionplot`. The bounds must be
aligned with `x`. Use `band` as the keyword-friendly constructor.
"""
struct Band{TX<:AbstractVector, TL<:AbstractVector, TU<:AbstractVector}
    x::TX
    lower::TL
    upper::TU
    label::String
    color::String
    alpha::Float64
end

function band(x::AbstractVector, lower::AbstractVector, upper::AbstractVector;
    label = "", color = "#D1495B", alpha::Real = 0.22)

    length(x) == length(lower) == length(upper) ||
        throw(DimensionMismatch("band coordinates must have equal length"))
    0 <= alpha <= 1 || throw(ArgumentError("band alpha must lie in [0, 1]"))
    return Band(x, lower, upper, String(label), string(color), Float64(alpha))
end

"""
    ReferenceLine(value; orientation = :vertical, kwargs...)

Store a dashed vertical or horizontal reference line for an exclusion or
function plot.
"""
struct ReferenceLine{T}
    value::T
    orientation::Symbol
    label::String
    color::String
    linestyle::Symbol
    linewidth::Float64
end

function reference_line(value; orientation::Symbol = :vertical, label = "",
    color = "#222222", linestyle::Symbol = :dash, linewidth::Real = 1.5)

    orientation in (:vertical, :horizontal) ||
        throw(ArgumentError("orientation must be :vertical or :horizontal"))
    return ReferenceLine(value, orientation, String(label), string(color), linestyle,
        Float64(linewidth))
end

"""
    PlotResult

Return value shared by the plotting pipeline. `figure` is the newly-created
figure when the renderer created one and `nothing` when an existing `axis` was
provided. `axis` and `plot` always refer to the rendered axis and primary plot.
"""
struct PlotResult{F, A, P}
    figure::F
    axis::A
    plot::P
end

"""Declare the optional CairoMakie box-and-whisker renderer."""
function boxplot end

"""Declare the optional CairoMakie scatter renderer."""
function scatterplot end

"""Declare the optional CairoMakie exclusion-curve renderer."""
function exclusionplot end

"""Declare the optional CairoMakie sampled-function renderer."""
function functionplot end

"""Declare the optional CairoMakie minima presentation renderer."""
function minima_plot end

"""Declare the optional CairoMakie trajectory renderer."""
function trajectoryplot end

"""Declare the optional CairoMakie figure-saving helper."""
function save_plot end

"""Declare the optional CairoMakie styled-axis constructor."""
function styled_axis end

"""Declare the legacy vacuum database plotters supplied by the extension."""
function vacua_db_jlm end
function vacua_db_jlm_single end
function total_geometries end
function vacua_db_jlm_box end

end
