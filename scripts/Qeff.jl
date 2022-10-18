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


@everywhere function main_Qsquare(h11,tri,cy,l)
    try
        id_basis = CYAxiverse.generate.vacua_id_basis(h11, tri, cy; threshold = 1e-2)
        if haskey(id_basis, "Qeff")
            return [0, h11, tri, cy]
        else
            return [1, h11, tri, cy]
        end
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

@everywhere function main_sortQ(h11list::Matrix)
    Qnon_square = []
    Qsquare = []
    for i in eachindex(h11list[1, :])
        if h11list[1, i] == 0
            push!(Qnon_square, h11list[:, i])
        else
            push!(Qsquare, h11list[:, i])
        end
    end
    if isfile(joinpath(CYAxiverse.filestructure.data_dir(),"Qshape.h5"))
        rm(joinpath(CYAxiverse.filestructure.data_dir(),"Qshape.h5"))
    end
    h5open(joinpath(CYAxiverse.filestructure.data_dir(),"Qshape.h5"), "cw") do f
        f["square",deflate=9] = hcat(Qsquare...)
        f["non_square",deflate=9] = hcat(Qnon_square...)
    end
end
lfile = CYAxiverse.filestructure.logfile()
CYAxiverse.filestructure.logcreate(lfile)

##############################
#### Initialise functions ####
##############################
@time temp_spec = main_Qsquare(4,10,1,lfile)
h11list_temp = [4 4 5 7; 10 11 10 10; 1 1 1 1; lfile lfile lfile lfile]
@time begin
    temp_spec = pmap(main_Qsquare, h11list_temp[1,:],h11list_temp[2,:],h11list_temp[3,:], h11list_temp[4,:])
end
@time main_sortQ(hcat(temp_spec...))
# println(temp_geom)
CYAxiverse.slurm.writeslurm(CYAxiverse.slurm.jobid,string((size(h11list_temp,2)+1), "test runs have finished.\n"))
### Clear memory ######
temp_spec = nothing
GC.gc()

##############################
############ Main ############
##############################
@time h11list = CYAxiverse.filestructure.paths_cy()[2]
CYAxiverse.slurm.writeslurm(CYAxiverse.slurm.jobid,string("There are ", size(h11list), "Qeff shapes to compute.\n"))
# h11 = shuffle(h11)
log_files_spec = [lfile for _=1:size(h11list,2)]
@time begin
    res = pmap(main_Qsquare,h11list[1,:],h11list[2,:],h11list[3,:],log_files_spec)
end
@time main_sortQ(hcat(res...))

GC.gc()
CYAxiverse.slurm.writeslurm(CYAxiverse.slurm.jobid,string("All workers are done!"))

