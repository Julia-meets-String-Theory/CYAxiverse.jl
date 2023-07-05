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
@everywhere using LinearAlgebra

@everywhere using HDF5
@everywhere using Random


@everywhere function main(geom_idx::CYAxiverse.structs.GeometryIndex,l::String)
    if isfile(CYAxiverse.filestructure.minfile(geom_idx))
        Nvac = 0
        h5open(CYAxiverse.filestructure.minfile(geom_idx), "r") do file
            Nvac = HDF5.read(file, "Nvac")
        end
        if Nvac == 0
            rm(CYAxiverse.filestructure.minfile(geom_idx))
            try
                res = CYAxiverse.jlm_minimizer.minimize_save(geom_idx)
                open(l, "a") do outf
                    write(outf,string("min-(",geom_idx.h11,",",geom_idx.polytope,",",geom_idx.frst,",\n"))
                end
            catch e
                open(l, "a") do outf
                    write(outf,string(stacktrace(catch_backtrace()),"--(",geom_idx.h11,",",geom_idx.polytope,",",geom_idx.frst,")\n"))
                end
            end
        end
    else
        try
            res = CYAxiverse.jlm_minimizer.minimize_save(geom_idx)
            open(l, "a") do outf
                write(outf,string("min-(",geom_idx.h11,",",geom_idx.polytope,",",geom_idx.frst,"),\n"))
            end
        catch e
            open(l, "a") do outf
                write(outf,string(stacktrace(catch_backtrace()),"--(",geom_idx.h11,",",geom_idx.polytope,",",geom_idx.frst,")\n"))
            end
        end
    end
end

lfile = CYAxiverse.filestructure.logfile()
CYAxiverse.filestructure.logcreate(lfile)

##############################
#### Initialise functions ####
##############################
geom_idx = CYAxiverse.structs.GeometryIndex(4, 10, 1)
@time temp_vac = main(geom_idx,lfile)
h11list_temp = [4 4 5 7; 10 11 10 10; 1 1 1 1]
h11list_temp = [CYAxiverse.structs.GeometryIndex(col...) for col in eachcol(h11list_temp)]
log_file_temp = [lfile for _ = 1:size(h11list_temp, 1)]
@time begin
    temp_vac = pmap(main, h11list_temp, log_file_temp)
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
h11list = h11list[:, h11list[1, :] .!= 491]
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
    res = pmap(main, geom_params, logfiles)
end

GC.gc()
CYAxiverse.slurm.writeslurm(CYAxiverse.slurm.jobid,string("All workers are done!"))

