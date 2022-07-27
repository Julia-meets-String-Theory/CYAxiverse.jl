# using Pkg
# Pkg.instantiate()

using Distributed
using HDF5, ArbNumerics, Distributions, Optim, LineSearches, Random
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

@everywhere function main(h11::Int,tri::Int,cy::Int,l::String,run_num::Int=1)
    prec = 1_000
    pot_data = CYAxiverse.read.potential(h11,tri,cy)
    QV::Matrix, LV::Matrix{Float64} = ArbFloat.(pot_data["Q"]), pot_data["L"]
    Lfull::Vector{ArbFloat} = ArbFloat.(LV[:,1]) .* ArbFloat(10.) .^ ArbFloat.(LV[:,2])
    gradσ = CYAxiverse.minimizer.grad_std(h11,tri,cy,Lfull,QV)
    h5open(CYAxiverse.filestructure.minfile(h11,tri,cy),isfile(CYAxiverse.filestructure.minfile(h11,tri,cy)) ? "r+" : "cw") do file
        if haskey(file, "gradσ")
        else
            f0 = create_group(file, "gradσ")
            f0["log10", deflate=9] = Float64.(log10.(gradσ))
        end
    end
    vac_data = CYAxiverse.generate.vacua_TB(pot_data["L"],pot_data["Q"])
    x0::Vector = ArbFloat.(rand(Uniform(0,2π),h11)) .* rand(ArbFloat,h11)##Do not declare type!  Breaks optimizer...
    algo_hz = Newton(alphaguess = LineSearches.InitialHagerZhang(α0=1.0), linesearch = LineSearches.HagerZhang())
    algo_LBFGS = LBFGS(linesearch = LineSearches.BackTracking())
    try
        res = CYAxiverse.minimizer.minimize_save(h11,tri,cy, Lfull, QV,x0,gradσ,vac_data["θ∥"],vac_data["Qtilde"],algo_LBFGS;prec=prec, run_num=run_num)
        open(l, "a") do outf
            write(outf,string("min-(",h11,",",tri,",",cy,",",run_num,")\n"))
        end
    catch e
        open(l, "a") do outf
            write(outf,string(stacktrace(catch_backtrace()),"--(",h11,",",tri,",",cy,",",run_num,")\n"))
        end
    end
end

lfile = CYAxiverse.filestructure.logfile()
CYAxiverse.filestructure.logcreate(lfile)

##############################
#### Initialise functions ####
##############################
@time temp_vac = main(4,10,1,lfile, 1)
h11list_temp = [4 4 5 7; 10 11 10 10; 1 1 1 1; lfile lfile lfile lfile; 1 2 1 1]
@time begin
    temp_vac = pmap(main, h11list_temp[1,:],h11list_temp[2,:],h11list_temp[3,:], h11list_temp[4,:], h11list_temp[5,:])
end
# println(temp_geom)
CYAxiverse.slurm.writeslurm(CYAxiverse.slurm.jobid,string((size(h11list_temp,2)+1), "test runs have finished.\n"))
### Clear memory ######
temp_vac = nothing
GC.gc()

##############################
############ Main ############
##############################

n=100#sample to minimize
x0i = 50#number of optimizations per geometry
split = round(Int,0.7*n)
geomparams = hcat(vcat(sort(rand(10:40,split)),sort(rand(40:100,n-split))), rand(1:100,n), ones(Int,n),)'
geomparams = geomparams[:,sortperm(geomparams[1,:])]
geomparams = [hcat([geomparams for _=1:x0i]...); vcat([ones(Int,n)*i for i=1:x0i]...)']
geomparams = geomparams[:, shuffle(1:end)]
ntasks = size(geomparams,2)
logfiles = [lfile for _=1:ntasks]

@time h11list = CYAxiverse.filestructure.paths_cy()[2]
CYAxiverse.slurm.writeslurm(CYAxiverse.slurm.jobid, "There are $ntasks random seeds to run on $np processors.\n")

@time begin
    res = pmap(main,geomparams[1,:],geomparams[2,:],geomparams[3,:],logfiles,geomparams[4,:])
end

GC.gc()
CYAxiverse.slurm.writeslurm(CYAxiverse.slurm.jobid,string("All workers are done!"))

