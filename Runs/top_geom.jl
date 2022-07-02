using Pkg
Pkg.instantiate()

using Distributed
# try
#     np = parse(Int32,ENV["SLURM_NPROCS"])
#     addprocs(np)
# catch e
#     error("no workers!")
#     exit()
# end
# @everywhere newARGS = string("vacua_new")

@everywhere using CYAxiverse

lfile = CYAxiverse.filestructure.logfile()
CYAxiverse.filestructure.logcreate(lfile)

@everywhere function main_top(h11,n,l)
    try
        test = CYAxiverse.generate.topologies(h11,n);
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

@everywhere function main_geom(h11,cy,tri,cy_i,l)
    try
        test = CYAxiverse.generate.geometries(h11,cy,tri,cy_i);
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
println(size(temp_top))
# println(temp_top)
@time temp_geom = pmap(main_geom,temp_top[1,:],temp_top[2,:],temp_top[3,:],temp_top[4,:], [lfile for _=1:size(temp_top,2)])
temp_geom = hcat(temp_geom...)
println(size(temp_geom))
# println(temp_geom)

### Clear memory ######
temp_top = nothing
temp_geom = nothing
GC.gc()

##############################
############ Main ############
##############################
h11_init = 4
np = nworkers()
h11_end = 200
log_files_top = [lfile for i=h11_init:h11_init+h11_end]
n = [100 for _=h11_init:h11_init+h11_end]
h11 = h11_init:h11_init+h11_end
writeslurm(jobid,string("There are ", size(h11), "topologies to run."))
# h11 = shuffle(h11)
@time begin
    h11cylist = pmap(main_top,h11,n,log_files_top)
end
# h11cylist = main_top(10,1000,lfile)
h11cylist = hcat(h11cylist...)
# h11cylist = h11cylist[:, shuffle(1:end)]
GC.gc()
writeslurm(jobid,string("There are ", size(h11cylist), "geometries to run."))
ntasks_cy = size(h11cylist,2)
log_files_geom = [lfile for _=1:ntasks_cy]
@time begin
    h11list = pmap(main_geom, h11cylist[1,:],h11cylist[2,:], h11cylist[3,:],h11cylist[4,:],log_files_geom)
end