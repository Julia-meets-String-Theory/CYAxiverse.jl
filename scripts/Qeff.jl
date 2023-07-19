# using Pkg
# Pkg.instantiate()

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

@everywhere using CYAxiverse
@everywhere using LinearAlgebra, Random

@everywhere using HDF5

@everywhere function extra_rows(h11, tri, cy, l)
    Qshape = CYAxiverse.read.qshape(h11, tri, cy)
    if Qshape.issquare == 1
    else
        data = CYAxiverse.generate.vacua_estimate(h11, tri, cy; threshold = 1e-2)
        h5open(joinpath(CYAxiverse.filestructure.geom_dir(h11,tri,cy),"qshape.h5"), "r+") do f
            f["extra_rows"] = data.extrarows
        end
    end
end

@everywhere function column_estimate(h11, tri, cy, l)
    geom_idx = CYAxiverse.structs.GeometryIndex(h11 = h11, polytope = tri, frst = cy)
    ωnorm = round(CYAxiverse.generate.ωnorm2(geom_idx; threshold = 0.01))
    h5open(joinpath(CYAxiverse.filestructure.geom_dir(h11,tri,cy),"qshape.h5"), "r+") do f
        if haskey(f, "ωnorm2_estimate")
        else
            f["ωnorm2_estimate"] = ωnorm
        end
    end
end

@everywhere function main_Qshape(geom_idx::CYAxiverse.structs.GeometryIndex, l)
    threshold = 1e-2
    h11, tri, cy = geom_idx.h11, geom_idx.polytope, geom_idx.frst
    try
        CYAxiverse.generate.vacua_estimate_save(geom_idx; threshold=threshold)
    catch e
        open(l, "a") do outf
            write(outf,string(stacktrace(catch_backtrace()),"--(",h11,",",tri,",",cy,")\n"))
        end
    finally
        open(l, "a") do outf
            write(outf,string("vac-(",h11,",",tri,",",cy,")\n"))
        end
    end
end


function main_sortQ(h11list::Matrix)
    Qnon_square = []
    Qsquare = []
    for col in eachcol(h11list)
        h11, tri, cy = col
        data = CYAxiverse.read.qshape(h11, tri, cy)
        square = data["issquare"]
        vac = data["vacua_estimate"]
        if square == 0 && vac != 0
            push!(Qnon_square, [h11 tri cy vac])
        elseif square == 1 && vac != 0
            push!(Qsquare, [h11 tri cy vac])
        end
    end
    if isfile(joinpath(CYAxiverse.filestructure.data_dir(),"Qshape.h5"))
        rm(joinpath(CYAxiverse.filestructure.data_dir(),"Qshape.h5"))
    end
    h5open(joinpath(CYAxiverse.filestructure.data_dir(),"Qshape.h5"), "cw") do f
        if !isa(Qsquare, Vector{Any})
            f["square",deflate=9] = hcat(Qsquare...)
        end
        if !isa(Qnon_square, Vector{Any})
            f["non_square",deflate=9] = hcat(Qnon_square...)
        end
    end
end

lfile = CYAxiverse.filestructure.logfile()
CYAxiverse.filestructure.logcreate(lfile)

##############################
#### Initialise functions ####
##############################
geom_idx = CYAxiverse.structs.GeometryIndex(4, 10, 1)
@time temp_vac = main_Qshape(geom_idx,lfile)
h11list_temp = [4 4 5 7; 10 11 10 10; 1 1 1 1]
h11list_temp = [CYAxiverse.structs.GeometryIndex(col...) for col in eachcol(h11list_temp)]
log_file_temp = [lfile for _ = 1:size(h11list_temp, 1)]
@time begin
    temp_vac = pmap(main_Qshape, h11list_temp, log_file_temp)
end
# println(temp_geom)
CYAxiverse.slurm.writeslurm(CYAxiverse.slurm.jobid,string((size(h11list_temp,2)+1), "test runs have finished.\n"))
### Clear memory ######
temp_vac = nothing
GC.gc()

##############################
############ Main ############
##############################
Random.seed!(1234567890)
h11list = CYAxiverse.filestructure.paths_cy()[2]
# h11list = h11list[:, h11list[1, :] .!= 491]
geom_params = [CYAxiverse.structs.GeometryIndex(col...) for col in eachcol(h11list)]
geom_params = shuffle!(geom_params)

##################################
##### Missing geoms ##############
##################################
# geom_params = geom_params[end-6_000:end, :]
##################################
ntasks = size(geom_params,1)
size_procs = size(np)
logfiles = [lfile for _=1:ntasks]

CYAxiverse.slurm.writeslurm(CYAxiverse.slurm.jobid, "There are $ntasks random seeds to run on $size_procs processors.\n")

@time begin
    res = pmap(main_Qshape, geom_params, logfiles)
end

GC.gc()
CYAxiverse.slurm.writeslurm(CYAxiverse.slurm.jobid,string("All workers are done!"))

