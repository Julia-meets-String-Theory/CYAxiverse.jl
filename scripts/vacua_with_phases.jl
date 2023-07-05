
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

@everywhere function optim_with_phases(geom_idx::CYAxiverse.structs.GeometryIndex, random_phase)
    CYAxiverse.jlm_minimizer.minimize_save(geom_idx, random_phase)
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
vac_square = hcat(vcat([item.square for item in vac_data]...)...)
vac_1D = Int.(hcat(vcat([item.one_dim for item in vac_data]...)...)[1:4, :])
vac_ND = Int.(hcat(vcat([item.n_dim for item in vac_data]...)...)[1:4, :])
no_vacua = hcat(vcat([item.err for item in vac_data]...)..., [item for item in eachcol(vac_ND) if item[4] == 0]...)
vac_ND = hcat([item for item in eachcol(vac_ND) if item[4] !=0]...)
all_vacua = hcat(vac_square, vac_1D, vac_ND)
println(size(all_vacua))
println(size(no_vacua))
GC.gc()
h11list = vcat(collect(4:332), [334, 336, 337, 338, 339, 340, 341, 345, 346, 347, 348, 350, 355, 357, 358, 366, 369, 370, 375, 376, 377, 386, 387, 399, 404, 416, 433, 462, 491])
n_full = size(CYAxiverse.filestructure.paths_cy()[2], 2)
n_list = [n_full for _ in 1:size(h11list, 1)]
@time begin
    vac_data = pmap(main, n_list, h11list)
end
vac_square = hcat(vcat([item.square for item in vac_data]...)...)
vac_1D = Int.(hcat(vcat([item.one_dim for item in vac_data]...)...)[1:4, :])
vac_ND = Int.(hcat(vcat([item.n_dim for item in vac_data]...)...)[1:4, :])
no_vacua = hcat(vcat([item.err for item in vac_data]...)..., [item for item in eachcol(vac_ND) if item[4] == 0]...)
vac_ND = hcat([item for item in eachcol(vac_ND) if item[4] !=0]...)
all_vacua = hcat(vac_square, vac_1D, vac_ND)

println(size(all_vacua))
println(size(no_vacua))
GC.gc()
@time large_vac = hcat([CYAxiverse.structs.GeometryIndex(item[1:3]...) for item in eachcol(all_vacua) if item[4] > 100])
println(size(large_vac))
random_phases = [true for _ in eachcol(large_vac)]
@time begin
    pmap(optim_with_phases, large_vac, random_phases)
end