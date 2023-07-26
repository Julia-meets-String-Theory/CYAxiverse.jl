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
@everywhere function is_subset_of(list1, list2)
    # Convert each vector in the lists to Set for efficient membership checking
    set_list1 = Set(list1)
    set_list2 = Set(list2)

    # Check if every vector in list1 is also present in list2
    for (i,vector) in enumerate(set_list1)
        if !(vector in set_list2)
            # return false, i
			return false
        end
    end

    return true
end
@everywhere function main(geom_idx::CYAxiverse.structs.GeometryIndex, l)
    h11, tri, cy = geom_idx.h11, geom_idx.polytope, geom_idx.frst
    try
        glsm = CYAxiverse.read.geometry(geom_idx)["glsm_charges"]
        qprime = CYAxiverse.read.potential(geom_idx).Q[1:h11, :]
        if is_subset_of(collect(eachrow(glsm)), collect(eachrow(qprime)))
        else
            open(l, "a") do outf
                write(outf,string("vac-(",h11,",",tri,",",cy,")\n"))
            end
        end
    catch e
        open(l, "a") do outf
            write(outf,string(stacktrace(catch_backtrace()),"--(",h11,",",tri,",",cy,")\n"))
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
    res = pmap(main, geom_params, logfiles)
end

GC.gc()
CYAxiverse.slurm.writeslurm(CYAxiverse.slurm.jobid,string("All workers are done!"))

