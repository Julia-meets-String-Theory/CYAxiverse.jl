# using Pkg
# Pkg.instantiate()

using Distributed
using HDF5
using MPIClusterManagers
import MPI
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


@everywhere function main_vac(h11,tri,cy,l)
    try
        test = CYAxiverse.generate.vacua_save_TB(h11,tri,cy);
    catch e
        open(l, "a") do outf
            write(outf,string(stacktrace(catch_backtrace()),"--(",h11,",",tri,",",cy,")\n"))
        end
    finally
        open(l, "a") do outf
            write(outf,string("vac-(",h11,",",tri,",",cy,")\n"))
        end
    end
    GC.gc()
end

lfile = CYAxiverse.filestructure.logfile()
CYAxiverse.filestructure.logcreate(lfile)

##############################
#### Initialise functions ####
##############################
@time temp_vac = main_vac(4,10,1,lfile)
h11list_temp = [4 4 5 7; 10 11 10 10; 1 1 1 1; lfile lfile lfile lfile]
@time begin
    temp_vac = pmap(main_vac, h11list_temp[1,:],h11list_temp[2,:],h11list_temp[3,:], h11list_temp[4,:])
end
# println(temp_geom)
CYAxiverse.slurm.writeslurm(CYAxiverse.slurm.jobid,string((size(h11list_temp,2)+1), "test runs have finished.\n"))
### Clear memory ######
temp_vac = nothing
GC.gc()

##############################
############ Main ############
##############################
@time h11list = CYAxiverse.filestructure.paths_cy()[2]
CYAxiverse.slurm.writeslurm(CYAxiverse.slurm.jobid,string("There are ", size(h11list), "systems to compute vacua in.\n"))
# h11 = shuffle(h11)
log_files_vac = [lfile for _=1:size(h11list,2)]
@time begin
    res = pmap(main_vac,h11list[1,:],h11list[2,:],h11list[3,:],log_files_vac)
end

GC.gc()
CYAxiverse.slurm.writeslurm(CYAxiverse.slurm.jobid,string("All workers are done!"))

