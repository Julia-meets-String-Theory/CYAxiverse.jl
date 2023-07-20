# using Pkg
# Pkg.instantiate()

using Distributed
# using MPIClusterManagers
# import MPI
# # MPI.initialize()
# manager = MPIClusterManagers.start_main_loop(MPI_TRANSPORT_ALL)
# # addprocs(manager)
# np = workers()
# println(np)
# if np!=0
# else
#     error("no workers!")
#     exit()
# end
try
    np = parse(Int32,ENV["SLURM_NPROCS"])
    addprocs(np, exeflags="--project=$(Base.active_project())")
catch e
    error("no workers!")
    exit()
end
split = nothing
if haskey(ENV, "SLURM_ARRAY_TASK_ID")
    split = parse(Int32, ENV["SLURM_ARRAY_TASK_ID"])
end

@everywhere using CYAxiverse
@everywhere using LinearAlgebra

@everywhere using HDF5
@everywhere using Random

@everywhere function is_subset_of(list1::Vector, list2::Vector)
    # Convert each vector in the lists to Set for efficient membership checking
    set_list1 = Set(list1)
    set_list2 = Set(list2)

    # Check if every vector in list1 is also present in list2
    for vector in set_list1
        if !(vector in set_list2)
            return false
        end
    end
    return true
end

@everywhere function main(geom_idx::CYAxiverse.structs.GeometryIndex,l::String)
    try
        Qtest = CYAxiverse.read.potential(geom_idx).Q[1:geom_idx.h11+4, :]
        hilbert_test = CYAxiverse.cytools_wrapper.hilbert_basis(Qtest)'
        Qtest = Qtest'
        if is_subset_of(collect(eachcol(hilbert_test)), collect(eachcol(Qtest)))
        else
            CYAxiverse.cytools_wrapper.hilbert_save(geom_idx, hilbert_test)
            open(l, "a") do outf
                write(outf,string("(",geom_idx.h11,",",geom_idx.polytope,",",geom_idx.frst,"),\n"))
            end
        end
    catch e
        open(l, "a") do outf
            write(outf,string(stacktrace(catch_backtrace()),"--(",geom_idx.h11,",",geom_idx.polytope,",",geom_idx.frst,")\n"))
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
# h11list = h11list[:, h11list[1, :] .!= 491]
# h11list = h11list[:, h11list[1, :] .== 225 .|| h11list[1, :] .== 249 .|| h11list[1, :] .== 252 .|| h11list[1, :] .== 254]
geom_params = [CYAxiverse.structs.GeometryIndex(col...) for col in eachcol(h11list)]
# geom_params = shuffle!(geom_params)

##################################
##### Missing geoms ##############
##################################
# geom_params = geom_params[end-6_000:end, :]
##################################
ntasks = size(geom_params,1)
size_procs = size(workers())
logfiles = [lfile for _=1:ntasks]

CYAxiverse.slurm.writeslurm(CYAxiverse.slurm.jobid, "There are $ntasks random seeds to run on $size_procs processors.\n")

@time begin
    res = pmap(main, geom_params, logfiles)
end

GC.gc()
CYAxiverse.slurm.writeslurm(CYAxiverse.slurm.jobid,string("All workers are done!"))

