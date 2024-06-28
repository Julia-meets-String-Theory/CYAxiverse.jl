module plotting
##############################
#### Plotting functions ######
##############################

using ..filestructure: plots_dir, count_geometries, paths_cy
using ..generate: jlm_vacua_db
using CairoMakie, ColorSchemes, Dates


"""
    vacua_db_jlm(n)

TBW
"""
function vacua_db_jlm(vac_data::NamedTuple)
    f = Figure()
    kwargs = (; xticklabelfont = "STIX", yticklabelfont = "STIX", xminorticksvisible = true, xminorgridvisible = true, yminorticksvisible = true, yminorgridvisible = true)
    ax1 = Axis(f[2, 1]; xticks = [4, 50, 100, 200, 300, 400, 491], xminorticks = IntervalsBetween(5), yscale = Makie.pseudolog10, kwargs...)
    ax2 = Axis(f[2, 2]; xticks = [4, 50, 100, 200, 300, 400, 491], xminorticks = IntervalsBetween(5), yscale = Makie.pseudolog10, yticks = [1, 10, 50, 100, 500, 1000, 2000], kwargs...)
    ax3 = Axis(f[1, 1:3]; xticks = [4, 50, 100, 200, 300, 400, 491], yscale = Makie.pseudolog10, kwargs...)
    square_vac = hcat([vcat(item[1], item[4]) for item in vac_data.square]...)
    n_dim_vac = hcat([vcat(item[1], item[4], item[end]) for item in vac_data.n_dim]...)
    n_dim_vac = n_dim_vac[:, sortperm(n_dim_vac[end, :])]
    one_dim_vac = hcat([vcat(item[1], item[4]) for item in vac_data.one_dim]...)
    sc_square = scatter!(ax3, square_vac[1, square_vac[2, :] .!= 0], square_vac[2, square_vac[2, :] .!= 0], marker = :circle, color = :green, markersize = 10, label = "0 extra rows")
    sc_onedim = scatter!(ax1, one_dim_vac[1, one_dim_vac[2, :] .!= 0], one_dim_vac[2, one_dim_vac[2, :] .!= 0], color = :cyan, marker = :utriangle, markersize = 10, label = "1 extra row")
    sc_ndim = scatter!(ax2, n_dim_vac[1, n_dim_vac[2, :] .!= 0], n_dim_vac[2, n_dim_vac[2, :] .!= 0], color = n_dim_vac[3, n_dim_vac[2, :] .!= 0], colormap = :thermal, marker = :rect, markersize = 10, label = L"$N$ extra rows")
    axislegend(ax3, [sc_square, sc_onedim, sc_ndim], [L"$0$", L"$1$", L"$N$"], "Number of Extra Rows", orientation = :horizontal, titlefont = "STIX Bold")
    Colorbar(f[2,3], limits = (minimum(n_dim_vac[3, n_dim_vac[2, :] .!= 0]), maximum(n_dim_vac[3, n_dim_vac[2, :] .!= 0])), colormap = :thermal, label = L"$N$", ticklabelfont = "STIX")
    Label(f[1:end, 0], L"$N_\mathrm{vacua}$", rotation = π/2)
    Label(f[end+1, 1:end], L"$h^{1,1}$")
    # save(joinpath(plots_dir(), "N_vac_KS.pdf"), f, pt_per_unit = 1)
    f
end
"""
    vacua_db_jlm(n)

TBW
"""
function vacua_db_jlm_single(vac_data::Matrix)
    size_inches = (60, 45)
    size_pt = 72 .* size_inches
    cmap = :bamako
    f = Figure(resolution = size_pt, fontsize=96, figure_padding = (5, 10, 5, 50))
    kwargs = (; xticklabelfont = "STIX", yticklabelfont = "STIX", xminorticksvisible = true, xminorgridvisible = true, yminorticksvisible = true, yminorgridvisible = true)
    ax1 = Axis(f[1, 1]; xticks = [1, 50, 100, 200, 300, 400, 491], yticks = [1, 10, 20, 30, 40, 54], xminorticks = IntervalsBetween(5), xlabel = L"$h^{1,1}$", ylabel = L"$N_\mathrm{vacua}$", kwargs...)
    vac_data = sortslices(vac_data, dims=2, by=x->x[4])
    all_vac = []
    for h11 in sort(collect(Set(vac_data[1, :])))
        for vac in sort(collect(Set(vac_data[4, :])))
			push!(all_vac, [h11, vac, size(vac_data[:, vac_data[1, :] .== h11 .&& vac_data[4, :] .== vac], 2)])
		end
	end
    xlims!(ax1, (-3, 494))
    all_vac = hcat(all_vac...)
    all_vac = all_vac[:, all_vac[3, :] .!= 0]
    scatter!(ax1, all_vac[1, :], all_vac[2, :], colormap = cmap, color = all_vac[3, :], marker = :rect, markersize = 25)
    Colorbar(f[1,2], colormap = cmap, limits = (minimum(all_vac[3, :]), maximum(all_vac[3, :])), labelpadding = 40, label = L"\text{No. of geometries}", ticks = [1, 200, 400, 600, 800, 1000], ticklabelfont = "STIX", size = 40)
    save(joinpath(plots_dir(), "N_vac_KS_scatter.pdf"), f, pt_per_unit = 1)
    f
end

function vacua_db_jlm(n=size(paths_cy()[2], 2); one_axis = false)
    vac_data = jlm_vacua_db(; n=n)
    vac_square = hcat([vcat(item[1:4]...) for item in vac_data.square]...)
	vac_1D = hcat([vcat(item[1:4]...) for item in vac_data.one_dim]...)
	vac_ND = hcat([vcat(item[1:4]..., item[end]...) for item in vac_data.n_dim]...)
	all_vacua = hcat(vac_square, vac_1D, vac_ND[1:4, :])
    println(size(all_vacua))
    if one_axis
        vacua_db_jlm_single(all_vacua)
    else
        vacua_db_jlm(vac_data)
    end
end


function total_geometries(n=nothing)
    total_geoms = count_geometries(n)
    fig_h11size = Figure()
	CairoMakie.barplot(fig_h11size[1,1], hcat(total_geoms...)[1,:], hcat(total_geoms...)[2,:])
    save(joinpath(plots_dir(), "total_geometries.pdf"), fig_h11size, pt_per_unit = 1)
end

"""
    vacua_db_jlm_box(square::Matrix, one_dim::Matrix, n_dim::Matrix)

TBW
"""
function vacua_db_jlm_box(square::Matrix, one_dim::Matrix, n_dim::Matrix; display = false, orientation = :horizontal)
	vacua_full = sortslices(hcat(square, one_dim, n_dim[1:4, :]), dims = 2, by=x->x[2])
    h11list = collect(Set(vacua_full[1, :]))
    max_h11 = min(400, maximum(h11list))
    min_h11 = minimum(h11list)
    if orientation == :horizontal
        size_inches = (36, 48)
    elseif orientation == :vertical
        size_inches = (48, 36)
    else
        error("Please specify boxplot orientation (options are :horizontal or :vertical)")
    end
	size_pt = 72 .* size_inches
    colors = resample_cmap(:twilight, maximum(h11list))
    f = Figure(resolution = size_pt, fontsize=56)
    kwargs1 = (; xticklabelfont = "STIX", yticklabelfont = "STIX", xminorticksvisible = true, xminorgridvisible = true, yminorticksvisible = true, yminorgridvisible = true, xlabelsize = 60, ylabelsize = 60, palette = (; patchcolor = colors[1:size(h11list[h11list .<= floor(max_h11 / 4)], 1)]))
    kwargs2 = (; xticklabelfont = "STIX", yticklabelfont = "STIX", xminorticksvisible = true, xminorgridvisible = true, yminorticksvisible = true, yminorgridvisible = true, xlabelsize = 60, ylabelsize = 60, palette = (; patchcolor = colors[size(h11list[h11list .<= floor(max_h11 / 4)], 1)+1:size(h11list[h11list .<= floor(max_h11 / 2)], 1)]))
    kwargs3 = (; xticklabelfont = "STIX", yticklabelfont = "STIX", xminorticksvisible = true, xminorgridvisible = true, yminorticksvisible = true, yminorgridvisible = true, xlabelsize = 60, ylabelsize = 60, palette = (; patchcolor = colors[size(h11list[h11list .<= floor(max_h11 / 2)], 1)+1:size(h11list[h11list .<= floor(3*max_h11 / 4)], 1)]))
    kwargs4 = (; xticklabelfont = "STIX", yticklabelfont = "STIX", xminorticksvisible = true, xminorgridvisible = true, yminorticksvisible = true, yminorgridvisible = true, xlabelsize = 60, ylabelsize = 60, palette = (; patchcolor = colors[size(h11list[h11list .<= floor(3*max_h11 / 4)], 1)+1 : end]))
    if orientation == :horizontal
        ax1 = Axis(f[1, 1]; kwargs1...)
        ax2 = Axis(f[1, 2]; kwargs2...)
        ax3 = Axis(f[1, 3]; kwargs3...)
        ax4 = Axis(f[1, 4]; kwargs4...)
    elseif orientation == :vertical
        ax1 = Axis(f[4, 1], xticks = [4, collect(20:20:100)...], xlabel = L"$N$"; kwargs1...)
        ax2 = Axis(f[3, 1], xticks = [collect(100:20:200)...]; kwargs2...)
        ax3 = Axis(f[2, 1], xticks = [collect(200:20:300)...]; kwargs3...)
        ax4 = Axis(f[1, 1], xticks = [collect(300:20:399)..., 404, 416, 433, 462, 491]; kwargs4...)
    end
    for item in sort(h11list[h11list .<= floor(max_h11 / 4)])
        CairoMakie.boxplot!(ax1, vacua_full[1, vacua_full[1, :] .== item], vacua_full[end, vacua_full[1, :] .== item], marker = :xcross, markersize = 10, whiskerwidth = 0.75, width = 0.9, orientation = orientation, gap = 0)
    end
    for item in sort(h11list[floor(max_h11 / 4) .< h11list .<= floor(max_h11 / 2)])
    	CairoMakie.boxplot!(ax2, vacua_full[1, vacua_full[1, :] .== item], vacua_full[end, vacua_full[1, :] .== item], marker = :xcross, markersize = 10, whiskerwidth = 0.75, width = 0.9, orientation = orientation, gap = 0)
	end
    for item in sort(h11list[floor(max_h11 / 2) .< h11list .<= floor(3*max_h11 / 4)])
        CairoMakie.boxplot!(ax3, vacua_full[1, vacua_full[1, :] .== item], vacua_full[end, vacua_full[1, :] .== item], marker = :xcross, markersize = 10, whiskerwidth = 0.75, width = 0.9, orientation = orientation, gap = 0)
    end
    for item in sort(h11list[floor(3*max_h11 / 4) .< h11list])
        CairoMakie.boxplot!(ax4, vacua_full[1, vacua_full[1, :] .== item], vacua_full[end, vacua_full[1, :] .== item], marker = :xcross, markersize = 10, whiskerwidth = 0.75, width = 0.9, orientation = orientation, gap = 0)
    end
    if orientation == :horizontal
        Colorbar(f[1:end, 0], limits=(4, maximum(h11list)), colormap = :twilight, ticklabelfont = "STIX", ticks = [4, collect(50:50:400)..., 491], nsteps = size(h11list, 1), label=L"$h^{1,1}$", flipaxis=false)
        Label(f[end+1, :],  L"$\mathcal{N}_{\!\!\mathrm{vac}}$")
        # Label(f[:, 0], L"$h^{1,1}$", rotation = π/2)
        for ax in [ax1, ax2, ax3, ax4]
            hideydecorations!(ax)
        end
        ylims!(ax1, 1, maximum(h11list[h11list .<= floor(max_h11 / 4)])+3)
        ylims!(ax2, minimum(h11list[floor(max_h11 / 4) .< h11list .<= floor(max_h11 / 2)]) - 3, maximum(h11list[floor(max_h11 / 4) .< h11list .<= floor(max_h11 / 2)])+3)
        ylims!(ax3, minimum(h11list[floor(max_h11 / 2) .< h11list .<= floor(3*max_h11 / 4)]) - 3, maximum(h11list[floor(max_h11 / 2) .< h11list .<= floor(3*max_h11 / 4)])+3)
        ylims!(ax4, minimum(h11list[floor(3*max_h11 / 4) .< h11list]) - 3, maximum(h11list[floor(3*max_h11 / 4) .< h11list])+3)
    elseif orientation == :vertical
        # Colorbar(f[7+1, 1:end], limits=(4, 103), colormap = colors[1:maximum(h11list[h11list .<= floor(max_h11 / 4)])], ticklabelfont = "STIX", ticks = [4, collect(10:10:100)...], nsteps = maximum(h11list[h11list .<= floor(max_h11 / 4)]), flipaxis=false, vertical = false, label=L"$N$")
        # Colorbar(f[5+1, 1:end], limits=(minimum(h11list[floor(max_h11 / 4) .< h11list .<= floor(max_h11 / 2)]), maximum(h11list[floor(max_h11 / 4) .< h11list .<= floor(max_h11 / 2)])+3), colormap = colors[minimum(h11list[floor(max_h11 / 4) .< h11list .<= floor(max_h11 / 2)]):maximum(h11list[floor(max_h11 / 4) .< h11list .<= floor(max_h11 / 2)])], ticklabelfont = "STIX", ticks = [collect(100:10:200)...], nsteps = (maximum(h11list[floor(max_h11 / 4) .< h11list .<= floor(max_h11 / 2)]) - minimum(h11list[floor(max_h11 / 4) .< h11list .<= floor(max_h11 / 2)])), flipaxis=false, vertical = false)
        # Colorbar(f[3+1, 1:end], limits=(minimum(h11list[floor(max_h11 / 2) .< h11list .<= floor(3*max_h11 / 4)]), maximum(h11list[floor(max_h11 / 2) .< h11list .<= floor(3*max_h11 / 4)])+3), colormap = colors[minimum(h11list[floor(max_h11 / 2) .< h11list .<= floor(3*max_h11 / 4)]):maximum(h11list[floor(max_h11 / 2) .< h11list .<= floor(3*max_h11 / 4)])], ticklabelfont = "STIX", ticks = [collect(200:10:300)...], nsteps = (maximum(h11list[floor(max_h11 / 2) .< h11list .<= floor(3*max_h11 / 4)]) - minimum(h11list[floor(max_h11 / 2) .< h11list .<= floor(3*max_h11 / 4)])), flipaxis=false, vertical = false)
        # Colorbar(f[1+1, 1:end], limits=(minimum(h11list[floor(3*max_h11 / 4) .< h11list]), maximum(h11list[floor(3*max_h11 / 4) .< h11list])+3), colormap = colors[minimum(h11list[floor(3*max_h11 / 4) .< h11list]):maximum(h11list[floor(3*max_h11 / 4) .< h11list])], ticklabelfont = "STIX", ticks = [collect(300:10:399)..., 404, 416, 433, 462, 491], nsteps = (maximum(h11list[floor(3*max_h11 / 4) .< h11list]) - minimum(h11list[floor(3*max_h11 / 4) .< h11list])), flipaxis=false, vertical = false)
        Label(f[:, 0],  L"$\mathcal{N}_{\!\!\mathrm{vac}}$", rotation = π/2)
        # Label(f[:, 0], L"$h^{1,1}$", rotation = π/2)
        # for ax in [ax1, ax2, ax3, ax4]
        #     hidexdecorations!(ax)
        # end
        xlims!(ax1, 1, 103)
        xlims!(ax2, 98, 203)
        xlims!(ax3, 198, 303)
        xlims!(ax4, 298, 493)
    end
    if display
        f
    else
        save(joinpath(plots_dir(), string(now(), "-N_vac_KS_box.pdf")), f, pt_per_unit = 1)
    end
end

function vacua_db_jlm_box(vacua_db::NamedTuple; display = false)
    square = hcat(vacua_db.square...)
	one_dim = hcat([item[1:4] for item in vacua_db.one_dim]...)
	n_dim = hcat([[item[1:4]...,item[end]] for item in vacua_db.n_dim if item[4] != 0]...)
	vacua_db_jlm_box(square, one_dim, n_dim; display = display)
end




end