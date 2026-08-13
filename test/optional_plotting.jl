using CairoMakie
using ColorSchemes

@testset "CairoMakie plotting pipeline" begin
    plotting = CYAxiverse.plotting
    @test Base.get_extension(CYAxiverse, :CYAxiverseCairoMakieExt) !== nothing

    style = plotting.paper_style(resolution = (480, 360), fontsize = 12)
    @test style.resolution == (480, 360)

    x = collect(range(0, 1; length = 32))
    y = x .^ 2
    lower = y .- 0.05
    upper = y .+ 0.05
    band = plotting.band(x, lower, upper; label = "95% C.L.")
    line = plotting.reference_line(0.5; label = "threshold")

    scatter = plotting.scatterplot(x, y; style = style, color = x, colorbar = true,
        colorbar_label = "sample")
    @test scatter.figure !== nothing
    @test scatter.axis !== nothing

    composed_figure = Figure(size = style.resolution)
    composed_axis = plotting.styled_axis(composed_figure[1, 1]; style)
    composed = plotting.scatterplot(x, y; style = style, axis = composed_axis)
    @test composed.figure === nothing
    @test composed.axis === composed_axis

    exclusion = plotting.exclusionplot(x, y; style = style, bands = (band,),
        reference_lines = (line,), xlabel = "x", ylabel = "fraction",
        label = "excluded", show_legend = true)
    @test exclusion.figure !== nothing

    boxes = plotting.boxplot([[1.0, 2.0, 3.0], [2.0, 4.0, 8.0]];
        style = style, labels = ["A", "B"])
    @test boxes.figure !== nothing

    function_result = plotting.functionplot(sin, x; style = style, label = "sin(x)")
    @test function_result.figure !== nothing

    minima = plotting.minima_plot(x, y; style = style, minima = [1, 32],
        critical = ([0.5], [0.25]))
    @test minima.plot.minimum !== nothing
    @test minima.plot.critical !== nothing

    trajectory = plotting.trajectoryplot(x, hcat(y, 1 .- y); style = style,
        labels = ["field 1", "field 2"], event_indices = [1, 32])
    @test trajectory.plot.events !== nothing

    mktempdir() do directory
        output = joinpath(directory, "exclusion.png")
        @test plotting.save_plot(output, exclusion) == output
        @test isfile(output)
    end
end
