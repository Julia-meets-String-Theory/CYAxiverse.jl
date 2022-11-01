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
@everywhere function main_Qshape(h11, tri, cy, l)
    threshold = 1e-2
    if isfile(joinpath(CYAxiverse.filestructure.geom_dir(h11,tri,cy),"qshape.h5"))
    else
        try
            CYAxiverse.generate.vacua_estimate_save(h11, tri, cy; threshold=threshold)
        catch e
            open(l, "a") do outf
                write(outf,string(stacktrace(catch_backtrace()),"--(",h11,",",tri,",",cy,")\n"))
            end
            if isfile(joinpath(CYAxiverse.filestructure.geom_dir(h11,tri,cy),"qshape.h5"))
                rm(joinpath(CYAxiverse.filestructure.geom_dir(h11,tri,cy),"qshape.h5"))
            end
            h5open(joinpath(CYAxiverse.filestructure.geom_dir(h11,tri,cy),"qshape.h5"), "cw") do f
                f["square", deflate=9] = 0
                f["vacua_estimate", deflate=9] = 0
            end
        finally
            open(l, "a") do outf
                write(outf,string("vac-(",h11,",",tri,",",cy,")\n"))
            end
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
@time temp_spec = main_Qshape(4,10,1,lfile)
h11list_temp = [4 4 5 7; 10 1 5 7; 1 1 1 1; lfile lfile lfile lfile]
@time begin
    temp_vac = pmap(main_Qshape, h11list_temp[1,:],h11list_temp[2,:],h11list_temp[3,:], h11list_temp[4,:])
end

@time begin
    main_sortQ(h11list_temp[1:3,:])
end
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
    res = pmap(main_Qshape,h11list[1,:],h11list[2,:],h11list[3,:],log_files_spec)
end

@time begin
    main_sortQ(h11list)
end


CYAxiverse.slurm.writeslurm(CYAxiverse.slurm.jobid,string("All workers are done!"))

