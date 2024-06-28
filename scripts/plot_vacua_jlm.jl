
using Distributed
import MPI
using MPIClusterManagers
# MPI.initialize()
manager = MPIClusterManagers.start_main_loop(MPI_TRANSPORT_ALL)
# addprocs(manager)
np = workers()
println(np)
if np!=0
else
    error("no workers!")
    exit()
end
ENV["newARGS"] = string("vacua_0323")

@everywhere using CYAxiverse
@everywhere using LinearAlgebra

@everywhere using HDF5
@everywhere using Random

@everywhere function main(n, h11)
	CYAxiverse.generate.jlm_vacua_db(; n=n, h11=h11)
end
@everywhere function random_phases_for_large_vacua(geom_idx::CYAxiverse.structs.GeometryIndex)
    try 
        CYAxiverse.jlm_minimizer.minimize_save(geom_idx; random_phase = true)
    catch e
        println([geom_idx.h11, geom_idx.polytope, geom_idx.frst])
    end
end
h11_temp = [4, 5, 6]
n10_temp = [10 for _ in 1:size(h11_temp, 1)]
n100_temp = [100 for _ in 1:size(h11_temp, 1)]
@time begin
    pmap(main, n10_temp, h11_temp)
end
@time begin
    pmap(main, n100_temp, h11_temp)
end
@time begin
    vac_data = pmap(main, [100 for _ in 4:40], collect(4:40))
end
vac_square = hcat(vcat([item.square for item in vac_data]...)...)[1:4, :]
vac_1D = Int.(hcat(vcat([item.one_dim for item in vac_data]...)...)[1:4, :])
vac_ND = Int.(hcat(vcat([item.n_dim for item in vac_data]...)...)[1:4, :])
no_vacua = hcat(vcat([item.err for item in vac_data]...)..., [item for item in eachcol(vac_ND) if item[4] == 0]...)
vac_ND = hcat([item for item in eachcol(vac_ND) if item[4] !=0]...)
all_vacua = hcat(vac_square, vac_1D, vac_ND, no_vacua)
large_vacua = all_vacua[:, all_vacua[4, :] .> 50]
println("Number of geometries collected:", size(all_vacua))
println("Number of geometries errored:", size(no_vacua))
println("Number of geometries with >10 vacua:", size(large_vacua))
# geom_params = [CYAxiverse.structs.GeometryIndex(col[1:3]...) for col in eachcol(no_vacua)]
# @time begin
#     pmap(random_phases_for_large_vacua, geom_params)
# end
# GC.gc()
# @time CYAxiverse.plotting.vacua_db_jlm_box(vac_square[:, vac_square[4, :] .< 100], vac_1D[:, vac_1D[4, :] .< 100], vac_ND[:, vac_ND[4, :] .< 100])
# GC.gc()
# h11list = vcat(collect(4:332), [334, 336, 337, 338, 339, 340, 341, 345, 346, 347, 348, 350, 355, 357, 358, 366, 369, 370, 375, 376, 377, 386, 387, 399, 404, 416, 433, 462, 491])
h11list = sort(collect(Set(CYAxiverse.filestructure.paths_cy()[2][1, :])))
n_list = [size(CYAxiverse.filestructure.paths_cy()[2][:, CYAxiverse.filestructure.paths_cy()[2][1, :] .== h11], 2) for h11 in h11list]
# n_list = [n_full for _ in 1:size(h11list, 1)]
@time begin
    vac_data = pmap(main, n_list, h11list)
end
vac_square = hcat(vcat([item.square for item in vac_data]...)...)[1:4, :]
vac_1D = Int.(hcat(vcat([item.one_dim for item in vac_data]...)...)[1:4, :])
vac_ND = Int.(hcat(vcat([item.n_dim for item in vac_data]...)...)[1:4, :])
# err_vacua = Int.(hcat(vcat([item.err for item in vac_data]...)...))
# no_vacua = hcat([item for item in eachcol(vac_ND) if item[4] ==0]...)
no_vacua = hcat(vcat([item.err for item in vac_data]...)..., [item for item in eachcol(vac_ND) if item[4] == 0]...)
all_vacua = hcat(vac_square, vac_1D, vac_ND, no_vacua)
large_vacua = all_vacua[:, all_vacua[4, :] .>= 20]
println("Number of geometries collected:", size(all_vacua))
println("Number of geometries errored:", size(no_vacua))
println("Number of geometries with >20 vacua:", size(large_vacua))
geom_params = [CYAxiverse.structs.GeometryIndex(col[1:3]...) for col in eachcol(no_vacua)]
@time begin
    pmap(random_phases_for_large_vacua, geom_params)
end
GC.gc()
@time begin
    vac_data = pmap(main, n_list, h11list)
end
GC.gc()
vac_square = hcat(vcat([item.square for item in vac_data]...)...)[1:4, :]
vac_1D = Int.(hcat(vcat([item.one_dim for item in vac_data]...)...)[1:4, :])
vac_ND = Int.(hcat(vcat([item.n_dim for item in vac_data]...)...)[1:4, :])
# err_vacua = Int.(hcat(vcat([item.err for item in vac_data]...)...))
# no_vacua = hcat([item for item in eachcol(vac_ND) if item[4] ==0]...)
no_vacua = hcat(vcat([item.err for item in vac_data]...)..., [item for item in eachcol(vac_ND) if item[4] == 0]...)
all_vacua = hcat(vac_square, vac_1D, vac_ND, no_vacua)
large_vacua = all_vacua[:, all_vacua[4, :] .>= 20]
println("Number of geometries collected:", size(all_vacua))
println("Number of geometries errored:", size(no_vacua))
println("Number of geometries with >20 vacua:", size(large_vacua))
@time begin
if isfile(joinpath(CYAxiverse.filestructure.data_dir(), "vacua_db.h5"))
    rm(joinpath(CYAxiverse.filestructure.data_dir(), "vacua_db.h5"))
end
h5open(joinpath(CYAxiverse.filestructure.data_dir(), "vacua_db.h5"), "cw") do file
    file["all_data", deflate=9] = all_vacua
end
end
@time begin
    CYAxiverse.plotting.vacua_db_jlm_single(all_vacua[:, all_vacua[4, :] .!= 0])
end
exit()