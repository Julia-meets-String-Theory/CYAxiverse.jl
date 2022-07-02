# using Pkg
# Pkg.instantiate()

using Distributed
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


@everywhere function main_spec(h11,tri,cy,l)
    try
        test = CYAxiverse.generate.pq_spectrum_save(h11,tri,cy);
    catch e
        open(l, "a") do outf
            write(outf,string(stacktrace(catch_backtrace()),"--(",h11,",",tri,",",cy,")\n"))
        end
    finally
        open(l, "a") do outf
            write(outf,string("spec-(",h11,",",tri,",",cy,")\n"))
        end
    end
end

lfile = CYAxiverse.filestructure.logfile()
CYAxiverse.filestructure.logcreate(lfile)

##############################
#### Initialise functions ####
##############################
@time temp_spec = main_spec(4,10,1,lfile)
h11list_temp = [4 4 5 7; 10 11 10 10; 1 1 1 1; lfile lfile lfile lfile]
@time begin
    temp_spec = pmap(main_spec, h11list_temp[1,:],h11list_temp[2,:],h11list_temp[3,:], h11list_temp[4,:])
end
# println(temp_geom)
CYAxiverse.slurm.writeslurm(CYAxiverse.slurm.jobid,string((size(h11list_temp,2)+1), "test runs have finished."))
### Clear memory ######
temp_spec = nothing
GC.gc()

##############################
############ Main ############
##############################
@time h11list = CYAxiverse.filestructure.paths_cy()[2]
CYAxiverse.slurm.writeslurm(CYAxiverse.slurm.jobid,string("There are ", size(h11list), "spectra to compute."))
# h11 = shuffle(h11)
log_files_spec = [lfile for _=1:size(h11list,2)]
@time begin
    res = pmap(main_spec,h11list[1,:],h11list[2,:],h11list[3,:],log_files_spec)
end

GC.gc()
CYAxiverse.slurm.writeslurm(CYAxiverse.slurm.jobid,string("All workers are done!"))

