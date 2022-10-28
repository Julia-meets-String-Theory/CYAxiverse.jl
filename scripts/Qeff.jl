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

@everywhere using HDF5
@everywhere function main_Qshape(h11, tri, cy, l)
    threshold = 1e-2
    try
        data = CYAxiverse.generate.LQtildebar(h11, tri, cy; threshold=threshold)
        if size(data["Qhat"], 1) == size(data["Qhat"], 2)
            vac = abs(det(data["Qhat"]))
            return Vector{Int}([1, h11, tri, cy, vac])
        else
            vac = floor(sqrt(abs(det(data["Qhat"] * data["Qhat"]'))))
            return Vector{Int}([0, h11, tri, cy, vac])
        end
    catch e
        open(l, "a") do outf
            write(outf,string(stacktrace(catch_backtrace()),"--(",h11,",",tri,",",cy,")\n"))
        end
        return [0, h11, tri, cy, 0]
    finally
        open(l, "a") do outf
            write(outf,string("vac-(",h11,",",tri,",",cy,")\n"))
        end
    end
end


@everywhere function main_sortQ(h11list::Matrix)
    Qnon_square = []
    Qsquare = []
    for i in eachindex(h11list[1, :])
        if h11list[1, i] == 0 && h11list[end, i] != 0
            push!(Qnon_square, h11list[2:end-1, i])
        elseif h11list[1, i] == 1 && h11list[end, i] != 0
            push!(Qsquare, h11list[2:end-1, i])
        end
    end
    println(Qnon_square)
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
h11list_temp = [4 4 5 7; 10 11 10 10; 1 1 1 1; lfile lfile lfile lfile]
@time begin
    temp_spec = pmap(main_Qshape, h11list_temp[1,:],h11list_temp[2,:],h11list_temp[3,:], h11list_temp[4,:])
end
println(temp_spec, typeof(temp_spec))
temp_spec = hcat(temp_spec...)
@time main_sortQ(temp_spec)

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
res = hcat(res...)
@time main_sortQ(res)


CYAxiverse.slurm.writeslurm(CYAxiverse.slurm.jobid,string("All workers are done!"))

