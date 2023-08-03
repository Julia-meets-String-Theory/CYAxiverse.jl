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
# @everywhere newARGS = string("vacua_new")

@everywhere using CYAxiverse
@everywhere using Random

lfile = CYAxiverse.filestructure.logfile()
CYAxiverse.filestructure.logcreate(lfile)



@everywhere function main_geom(geom_idx::CYAxiverse.structs.GeometryIndex,l)
    try
        CYAxiverse.cytools_wrapper.geometries_hilbert(geom_idx);
        open(l, "a") do outf
            write(outf,string("geom-(",geom_idx.h11,",",geom_idx.polytope,",",geom_idx.frst,")\n"))
        end
    catch e
        open(l, "a") do outf
            write(outf,string(stacktrace(catch_backtrace()),"--(",geom_idx.h11,",",geom_idx.polytope,",",geom_idx.frst,")\n"))
        end
    end
    
end




##############################
#### Initialise functions ####
##############################
@time main_geom(CYAxiverse.structs.GeometryIndex(4, 1, 1), lfile)
h11list_temp = [4 4 5 7; 10 11 10 10; 1 1 1 1]
h11list_temp = [CYAxiverse.structs.GeometryIndex(col...) for col in eachcol(h11list_temp)]
log_file_temp = [lfile for _ = 1:size(h11list_temp, 1)]

# temp_top = hcat(temp_top...)
# println(size(temp_top))
# println(temp_top)

@time temp_geom = pmap(main_geom,h11list_temp, log_file_temp)
# println(size(temp_geom))
# println(temp_geom)
CYAxiverse.slurm.writeslurm(CYAxiverse.slurm.jobid,string(size(h11list_temp, 2), "test runs have finished.\n"))
### Clear memory ######
temp_top = nothing
temp_geom = nothing
GC.gc()
# all_h11 = vcat(collect(4:332), [334, 336, 337, 338, 339, 340, 341, 345, 346, 347, 348, 350, 355, 357, 358, 366, 369, 370, 375, 376, 377, 386, 387, 399, 404, 416, 433, 462, 491])
##############################
############ Main ############
##############################
Random.seed!(1234567890)
h11list = CYAxiverse.filestructure.paths_cy()[2]
# h11list = h11list[:, h11list[1, :] .!= 491]
h11list = h11list[:, h11list[1, :] .<= 55]
geom_params = [CYAxiverse.structs.GeometryIndex(col...) for col in eachcol(h11list)]
geom_params = shuffle!(geom_params)

##################################
##### Missing geoms ##############
##################################
# geom_params = geom_params[end-6_000:end, :]
##################################
ntasks = size(geom_params,1)
size_procs = nworkers()
logfiles = [lfile for _=1:ntasks]

CYAxiverse.slurm.writeslurm(CYAxiverse.slurm.jobid, "There are $ntasks random seeds to run on $size_procs processors.\n")

@time begin
    res = pmap(main_geom, geom_params, logfiles)
end

GC.gc()
CYAxiverse.slurm.writeslurm(CYAxiverse.slurm.jobid,string("All workers are done!"))

