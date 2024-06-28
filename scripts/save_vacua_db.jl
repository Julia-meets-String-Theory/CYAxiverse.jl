
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

function save_vacua_db(square::Matrix, one_dim::Matrix, n_dim::Matrix)
    if isfile(joinpath(CYAxiverse.filestructure.data_dir(),"vacua_jlm_db.h5"))
        rm(joinpath(CYAxiverse.filestructure.data_dir(),"vacua_jlm_db.h5"))
    end
    h5open(joinpath(CYAxiverse.filestructure.data_dir(),"vacua_jlm_db.h5"), "cw") do f
        f["square", deflate=9] = square
        f["one_dim", deflate=9] = one_dim
        f["n_dim", deflate=9] = n_dim
        # f["detQtilde", deflate=9] = detQtilde
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
# vac_detQ = hcat(vcat([item.detQ for item in vac_data]...)...)
vac_1D = Int.(hcat(vcat([item.one_dim for item in vac_data]...)...)[1:4, :])
vac_ND = Int.(hcat(vcat([item.n_dim for item in vac_data]...)...)[1:4, :])
no_vacua = hcat(vcat([item.err for item in vac_data]...)..., [item for item in eachcol(vac_ND) if item[4] == 0]...)
vac_ND = hcat([item for item in eachcol(vac_ND) if item[4] !=0]...)
all_vacua = hcat(vac_square, vac_1D, vac_ND)
large_vacua = all_vacua[:, all_vacua[4, :] .> 100]
println(size(all_vacua))
println(size(no_vacua))
println(size(large_vacua))
@time save_vacua_db(vac_square, vac_1D, vac_ND)
GC.gc()
h11list = vcat(collect(4:332), [334, 336, 337, 338, 339, 340, 341, 345, 346, 347, 348, 350, 355, 357, 358, 366, 369, 370, 375, 376, 377, 386, 387, 399, 404, 416, 433, 462, 491])
n_full = size(CYAxiverse.filestructure.paths_cy()[2], 2)
n_list = [n_full for _ in 1:size(h11list, 1)]
@time begin
    vac_data = pmap(main, n_list, h11list)
end
vac_square = hcat(vcat([item.square for item in vac_data]...)...)[1:4, :]
# vac_detQ = hcat(vcat([item.detQ for item in vac_data]...)...)
vac_1D = Int.(hcat(vcat([item.one_dim for item in vac_data]...)...)[1:4, :])
vac_ND = Int.(hcat(vcat([item.n_dim for item in vac_data]...)...)[1:4, :])
err_vacua = Int.(hcat(vcat([item.err for item in vac_data]...)...))
no_vacua = hcat([item for item in eachcol(vac_ND) if item[4] == 0]...)
vac_ND = hcat([item for item in eachcol(vac_ND) if item[4] !=0]...)
all_vacua = hcat(vac_square, vac_1D, vac_ND)
large_vacua = all_vacua[:, all_vacua[4, :] .> 100]
println(size(all_vacua))
println(size(err_vacua))
println(size(no_vacua))
println(size(large_vacua))
GC.gc()
@time save_vacua_db(vac_square, vac_1D, vac_ND)