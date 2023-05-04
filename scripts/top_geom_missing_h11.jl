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
@everywhere function main_top_fair(h11,n,l)
    try
        test = CYAxiverse.cytools_wrapper.cy_from_poly(h11);
        return test
    catch
        try
            test = CYAxiverse.cytools_wrapper.topologies(h11,n; fast=false);
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
if split == nothing
    @time temp_top = main_top(Random.rand(4:10),10,lfile)
else 
    @time temp_top = main_top(max(Random.rand(4:10),split),10,lfile)
end
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
#### Find incomplete h11s ####
##############################
function h11_list(h11 ;geometric_data = true)
    try
        h11list = CYAxiverse.filestructure.np_path_generate(h11; geometric_data= geometric_data)[2]
        if size(h11list, 1) !=0
            return h11list
        else
            return [h11, 0, 0]
        end
    catch e
        open(l, "a") do outf
            write(outf,string(stacktrace(catch_backtrace()),"\n (",h11,")"))
        end
        return [h11, 0, 0]
    end
end
h11_full = vcat(collect(4:332), [334, 336, 337, 338, 339, 340, 341, 345, 346, 347, 348, 350, 355, 357, 358, 366, 369, 370, 375, 376, 377, 386, 387, 399, 404, 416, 433, 462, 491])
function h11_missing(h11list::Vector)
    h11 = []
    for poly in h11_full
        temp_h11 = h11_list(h11)
        if temp_h11[2] == 0
            push!(h11, poly)    
        end
    end
    h11
end
h11 = [223, 226, 228, 235, 249, 250, 252, 253, 254, 255, 256, 257, 258]

##############################
############ Main ############
##############################
np = nworkers()
max_split = 0
n_split = 1
if haskey(ENV, "MAX_JOB")
    max_split = parse(Int32, ENV["MAX_JOB"])
end
if haskey(ENV, "SLURM_ARRAY_TASK_COUNT")
    n_split = parse(Int32, ENV["SLURM_ARRAY_TASK_COUNT"])
end


function h11list_generate(h11::Vector, lfile::String; ngeometries::Int = 10, split = nothing, max_split = 0, n_split = 1)
    log_files_top = []
    n = []
    if split === nothing
        log_files_top = [lfile for _ in h11]
        n = [ngeometries for _ in h11]
    else
        if split == max_split
            h11 = [462, 491]
            n = [ngeometries for _ in h11]
            log_files_top = [lfile for _ in h11]
            
        else
            Random.seed!(9876543210)
            h11 = shuffle(h11)
            tasks = length(h11) ÷ n_split
            h11 = sort(h11[(split - 1) * tasks + 1 : split * tasks])
            n = [ngeometries for _ in h11]
            log_files_top = [lfile for _ in h11]
        end
    end
    (h11 = h11, log_files = log_files_top, ngeometries = n)
end

run_vars = h11list_generate(h11, lfile; ngeometries=1_000, split=split, max_split = max_split, n_split = n_split)

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