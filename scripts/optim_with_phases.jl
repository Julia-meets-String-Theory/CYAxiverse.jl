
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
ENV["newARGS"] = string("vacua_0323")

@everywhere using CYAxiverse
@everywhere using LinearAlgebra

@everywhere using HDF5
@everywhere using Random

@everywhere function main(n, h11)
	try
        CYAxiverse.generate.jlm_vacua_db(; n=n, h11=h11)
    catch e
        println(h11)
    end
end

@everywhere function optim_with_phases(geom_idx::CYAxiverse.structs.GeometryIndex)
    try
        CYAxiverse.jlm_minimizer.minimize_save(geom_idx; random_phase = true)
        open(l, "a") do outf
            write(outf,string("min-(",geom_idx.h11,",",geom_idx.polytope,",",geom_idx.frst,",\n"))
        end
    catch e
        open(l, "a") do outf
            write(outf,string(stacktrace(catch_backtrace()),"--(",geom_idx.h11,",",geom_idx.polytope,",",geom_idx.frst,")\n"))
        end
    end
    
end

h11list = CYAxiverse.filestructure.paths_cy()[2]

