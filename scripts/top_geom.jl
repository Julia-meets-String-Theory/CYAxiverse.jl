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
# @everywhere newARGS = string("vacua_new")

@everywhere using CYAxiverse
@everywhere using Random

lfile = CYAxiverse.filestructure.logfile()
CYAxiverse.filestructure.logcreate(lfile)

@everywhere function main_top(h11,n,l)
    try
        test = CYAxiverse.cytools_wrapper.cy_from_poly(h11);
        return test
    catch
        try
            test = CYAxiverse.cytools_wrapper.topologies(h11,n);
            return test
        catch e
            open(l, "a") do outf
                write(outf,string(stacktrace(catch_backtrace()),"\n (",h11,")"))
            end
            return [0,0,0,0]
        finally
            open(l, "a") do outf
                write(outf,string("top-(",h11,")\n"))
            end
        end
    end
    
end

@everywhere function main_geom(h11,cy,tri,cy_i,l)
    try
        test = CYAxiverse.cytools_wrapper.geometries(h11,cy,tri,cy_i);
        return test
    catch e
        open(l, "a") do outf
            write(outf,string(stacktrace(catch_backtrace()),"--(",h11,",",tri,",",cy_i,")\n"))
        end
        return [0,0,0]
    finally
        open(l, "a") do outf
            write(outf,string("geom-(",h11,",",tri,",",cy_i,")\n"))
        end
    end
    
end




##############################
#### Initialise functions ####
##############################
@time temp_top = main_top(4,10,lfile)
# temp_top = hcat(temp_top...)
# println(size(temp_top))
# println(temp_top)

@time temp_geom = pmap(main_geom,temp_top[1,:],temp_top[2,:],temp_top[3,:],temp_top[4,:], [lfile for _=1:size(temp_top,2)])
temp_geom = hcat(temp_geom...)
# println(size(temp_geom))
# println(temp_geom)
CYAxiverse.slurm.writeslurm(CYAxiverse.slurm.jobid,string(size(temp_geom), "test runs have finished.\n"))
### Clear memory ######
temp_top = nothing
temp_geom = nothing
GC.gc()

##############################
############ Main ############
##############################
h11_init = 4
np = nworkers()
h11_end = 500
h11 = collect(h11_init:h11_init+h11_end)
max_split = 0
if haskey(ENV, "MAX_JOB")
    max_split = parse(Int32, ENV["MAX_JOB"])
end

function h11list_generate(h11::Vector, lfile::String; ngeometries::Int = 10, split = nothing, max_split = 0)
    log_files_top = []
    n = []
    if split === nothing
        log_files_top = [lfile for _ in h11]
        n = [ngeometries for _ in h11]
    else
        if split == max_split
            h11 = [462, 491]
            n = [ngeometries * 1_000 for _ in h11]
            log_files_top = [lfile for _ in h11]
            
        else
            Random.seed!(9876543210)
            h11 = shuffle(h11)
            tasks = length(h11) ÷ max_split
            h11 = sort(h11[(split - 1) * tasks + 1 : split * tasks])
            n = [ngeometries * 1_000 for _ in h11]
            log_files_top = [lfile for _ in h11]
        end
    end
    (h11 = h11, log_files = log_files_top, ngeometries = n)
end

run_vars = h11list_generate(h11, lfile; ngeometries=10, split=split, max_split = max_split)

h11 = run_vars.h11
n = run_vars.ngeometries
log_files_top = run_vars.log_files

CYAxiverse.slurm.writeslurm(CYAxiverse.slurm.jobid,string("There are ", size(h11), "topologies to run.\n"))
CYAxiverse.slurm.writeslurm(CYAxiverse.slurm.jobid,string("These are ", h11, "\n"))
@time begin
    h11cylist = pmap(main_top,h11,n,log_files_top)
end

h11cylist = hcat(h11cylist...)[:, hcat(h11cylist...)[1,:] .!= 0]
# h11cylist = h11cylist[:, shuffle(1:end)]

GC.gc()

CYAxiverse.slurm.writeslurm(CYAxiverse.slurm.jobid,string("There are ", size(h11cylist), "geometries to run.\n"))

ntasks_cy = size(h11cylist,2)
log_files_geom = [lfile for _=1:ntasks_cy]
@time begin
    h11list = pmap(main_geom, h11cylist[1,:],h11cylist[2,:], h11cylist[3,:],h11cylist[4,:],log_files_geom)
end