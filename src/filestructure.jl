"""
    CYAxiverse.filestructure
This module sets up the structure of the database, identifying where to locate data / plot files etc

"""
module filestructure
#######################
### Test functions ####
########################
using HDF5
using Dates

###############################
### Initialising functions ####
###############################
"""
    localARGS()
Load key for data dir -- key should be in ol_DB
"""
function localARGS()
    if haskey(ENV,"newARGS")
        newARGS = ENV["newARGS"]
    else
        ARGS
    end
end
"""
    ol_DB(args)

Define dict of directories for data read/write
"""
function ol_DB(args)
    ol_db::Dict{String,String} = Dict(
        "KU_Fair" => "/home/uni09/cosmo/mehta2/KSAxiverse_Jun20_InKC/KSAxiverse_KU_Fair_Large/",
        "inKC" => "/home/uni09/cosmo/mehta2/KSAxiverse_Jun20_InKC/KSAxiverse_Scaled/", 
        "home_Large" => "/home/uni09/cosmo/mehta2/KSAxiverse_Jun20_InKC/KSAxiverse/", 
        "KU1" => "/scratch/Axiverse_Learning/KU1/", 
        "KV1" => "/scratch/Axiverse_Learning/KV1/",
        "KV25" => "/scratch/Axiverse_Learning/KV25/", 
        "vacua_test" => "/scratch/users/mehta2/vacua_testing/", 
        "vacua_new" => "/scratch/users/mehta2/vacua_db/",
        "vacua_0822" => "/scratch/users/mehta2/vacua_0822/",
        "vacua_stretch" => "/scratch/users/mehta2/vacua_stretch/",
        "pwd" => string(pwd(), "/")
        )
    try
        ol_db[string(args)]
    catch y
        ol_db["pwd"]
    end
end

############################
### Directory functions ####
############################
"""
    present_dir()

Returns the present data directory using localARGS
"""
function present_dir()
    pwd = ol_DB(localARGS())
    return pwd
end
"""
    plots_dir()

Creates/reads a directory for plots
"""
function plots_dir()
    pwd = string(ol_DB(localARGS()),"plots")
    if isdir(pwd)
    else
        mkpath(pwd)
    end
    return pwd
end
"""
    log_dir()
Creates/reads log directory
"""
function log_dir()
    if isdir(joinpath(present_dir(),"logs"))
    else 
        mkdir(joinpath(present_dir(),"logs"))
    end
    return joinpath(present_dir(),"logs")
end

"""
    data_dir()
Creates/reads data directory
"""
function data_dir()
    if isdir(joinpath(present_dir(),"data"))
    else 
        mkdir(joinpath(present_dir(),"data"))
    end
    return joinpath(present_dir(),"data")
end
"""
    logfile()
Returns path of logfile in format data_dir()/logs/YYYY:MM:DD:T00:00:00.000log.out
"""
function logfile()
    log = string(Dates.DateTime(Dates.now()),"log.out")
    return joinpath(log_dir(), log)
end

"""
    logcreate(l)
Creates logfile
"""
function logcreate(l::String)
    open(l, "w") do outf
        write(outf,string(Dates.DateTime(Dates.now()),"\n"))
    end
end

"""
    np_path()
Walks through `data_dir()` and returns list of data paths and matrix of `[h11; tri; cy]`.
Saves in h5 file `paths_cy.h5`
"""
function np_path()
    np_paths = Vector{UInt8}[]
    np_pathinds = Vector{Int}[]
    for i in first(walkdir(present_dir()))[2]
        if occursin(r"h11_*", i)
            for j in first(walkdir(joinpath(present_dir(),i)))[2]
                if occursin(r"np_*", j)
                    for k in first(walkdir(joinpath(present_dir(),i,j)))[2]
                        if occursin(r"cy_*", k)
                            if isfile(joinpath(present_dir(),i,j,k,"cyax.h5"))
                                push!(np_paths,transcode(UInt8,joinpath(i,j,k)))
                                push!(np_pathinds,[parse(Int,SubString(i,5,7)),parse(Int,SubString(j,4,10)),parse(Int,SubString(k,4,10))])
#                                 println([parse(Int,SubString(i,5,7)),parse(Int,SubString(j,4,10)),parse(Int,SubString(k,4,10))])
                            end
                        end
                    end
                end
            end
        end
    end
    if isfile(joinpath(data_dir(),"paths.h5")) || isfile(joinpath(data_dir(),"paths_cy.h5"))
    else
        h5open(joinpath(data_dir(),"paths_cy.h5"), "cw") do f
            f["paths",deflate=9] = hcat(np_paths...)
            f["pathinds",deflate=9] = hcat(np_pathinds...)
        end
    end
    return hcat(np_paths...), hcat(np_pathinds...)
end
"""
    paths_cy()
Loads / generates `paths_cy.h5` which contains the explicit locations and also `[h11; tri; cy]` indices of the geometries already saved.
"""
function paths_cy()
    if isfile(joinpath(data_dir(),"paths.h5")) || isfile(joinpath(data_dir(),"paths_cy.h5"))
    else
        np_path()
    end
    if localARGS()==string("in_KC") 
        paths_cy,pathinds_cy =  h5open(joinpath(data_dir(),"paths.h5"), "r") do f
            read(f,"paths"),read(f,"pathinds")
            end;
    else
        paths_cy,pathinds_cy =  h5open(joinpath(data_dir(),"paths_cy.h5"), "r") do f
            read(f,"paths"),read(f,"pathinds")
            end;
    end
    if typeof(paths_cy) == Matrix{UInt8}
        paths_cy = [transcode(String,paths_cy[:,i]) for i in 1:size(paths_cy,2)]
    end
    return paths_cy,pathinds_cy
end
#######################
### Misc functions ####
#######################
"""
    h11lst(min,max)
Loads geometry indices between ``h^{1,1} \\in (\\mathrm{min},\\mathrm{max}]``
"""
function h11lst(h11min=0,h11max=100)
    pathinds_cy = paths_cy()[2]
    h11list = pathinds_cy[:,h11min .< pathinds_cy[1,:].<= h11max]
    return h11list
end

"""
    geom_dir(h11,tri,cy)
Defines file directories for data specified by geometry index.
"""
function geom_dir(h11,tri,cy=1)
    if localARGS()!=string("inKC")
        if localARGS()==string("home_Large")||localARGS()==string("KV1")
            if h11 >= 238
                if isdir(string(present_dir(),"h11_",lpad(h11,3,"0"),"/np_",lpad(tri,7,"0"),"/cy_",lpad(cy,7,"0")))
                    string(present_dir(),"h11_",lpad(h11,3,"0"),"/np_",lpad(tri,7,"0"),"/cy_",lpad(cy,7,"0"))
                else
                    mkdir(string(present_dir(),"h11_",lpad(h11,3,"0"),"/np_",lpad(tri,7,"0"),"/cy_",lpad(cy,7,"0")))
                end
            else
                if isdir(string(present_dir(),"h11_",lpad(h11,3,"0"),"/np_",lpad(tri,7,"0")))
                    string(present_dir(),"h11_",lpad(h11,3,"0"),"/np_",lpad(tri,7,"0"))
                else
                    mkdir(string(present_dir(),"h11_",lpad(h11,3,"0"),"/np_",lpad(tri,7,"0")))
                end
            end
        else
            if isdir(string(present_dir(),"h11_",lpad(h11,3,"0"),"/np_",lpad(tri,7,"0"),"/cy_",lpad(cy,7,"0")))
                    string(present_dir(),"h11_",lpad(h11,3,"0"),"/np_",lpad(tri,7,"0"),"/cy_",lpad(cy,7,"0"))
            else
                mkdir(string(present_dir(),"h11_",lpad(h11,3,"0"),"/np_",lpad(tri,7,"0"),"/cy_",lpad(cy,7,"0")))
            end
        end
    else
        if isdir(string(present_dir(),"h11_",lpad(h11,3,"0"),"/np_",lpad(tri,7,"0")))
            string(present_dir(),"h11_",lpad(h11,3,"0"),"/np_",lpad(tri,7,"0"))
        else
            mkdir(string(present_dir(),"h11_",lpad(h11,3,"0"),"/np_",lpad(tri,7,"0")))
        end
    end
end

###################################
### Geometric Data Files (old) ####
###################################
"""
    Kfile(h11,tri,cy)
Loads Kähler metric specified by geometry index.
!!! warning
    Deprecated
"""
function Kfile(h11,tri, cy=1)
    if localARGS()!=string("inKC")
        if localARGS()==string("home_Large")||localARGS()==string("KV1")
            if h11 >= 238
                string(present_dir(),"h11_",lpad(h11,3,"0"),"/np_",lpad(tri,7,"0"),"/cy_",lpad(cy,7,"0"),"/K.hdf5")
            else
                string(present_dir(),"h11_",lpad(h11,3,"0"),"/np_",lpad(tri,7,"0"),"/K.hdf5")
            end
        else
            string(present_dir(),"h11_",lpad(h11,3,"0"),"/np_",lpad(tri,7,"0"),"/cy_",lpad(cy,7,"0"),"/K.hdf5")
        end
    else
        string(present_dir(),"h11_",lpad(h11,3,"0"),"/np_",lpad(tri,7,"0"),"/K.hdf5")
    end
end
"""
    Qfile(h11,tri,cy)
Loads instanton charge matrix specified by geometry index.
!!! warning
    Deprecated
"""
function Qfile(h11,tri, cy=1)
    if localARGS()!=string("inKC") 
        if localARGS()==string("home_Large")||localARGS()==string("KV1")
            if h11 >= 238
                string(present_dir(),"h11_",lpad(h11,3,"0"),"/np_",lpad(tri,7,"0"),"/cy_",lpad(cy,7,"0"),"/Q.hdf5")
            else
                string(present_dir(),"h11_",lpad(h11,3,"0"),"/np_",lpad(tri,7,"0"),"/Q.hdf5")
            end
        else
            string(present_dir(),"h11_",lpad(h11,3,"0"),"/np_",lpad(tri,7,"0"),"/cy_",lpad(cy,7,"0"),"/Q.hdf5")
        end
    else
        string(present_dir(),"h11_",lpad(h11,3,"0"),"/np_",lpad(tri,7,"0"),"/Q.hdf5")
    end
end
"""
    Lfile(h11,tri,cy)
Loads instanton energy scales specified by geometry index.
!!! warning
    Deprecated
"""
function Lfile(h11,tri, cy=1)
    if localARGS()!=string("inKC") 
        if localARGS()==string("home_Large")||localARGS()==string("KV1")
            if h11 >= 238
                string(present_dir(),"h11_",lpad(h11,3,"0"),"/np_",lpad(tri,7,"0"),"/cy_",lpad(cy,7,"0"),"/L.hdf5")
            else
                string(present_dir(),"h11_",lpad(h11,3,"0"),"/np_",lpad(tri,7,"0"),"/L.hdf5")
            end
        else
            string(present_dir(),"h11_",lpad(h11,3,"0"),"/np_",lpad(tri,7,"0"),"/cy_",lpad(cy,7,"0"),"/L.hdf5")
        end
    else
        string(present_dir(),"h11_",lpad(h11,3,"0"),"/np_",lpad(tri,7,"0"),"/L.hdf5")
    end
end

###################
### Data Files ####
###################
"""
    cyax_file(h11,tri,cy)
Path to data file -- will contain all data that relates to geometry index.
"""
function cyax_file(h11,tri, cy=1)
    return string(geom_dir(h11,tri,cy),"/cyax.h5")
end
"""
    minfile(h11,tri,cy)
Path to file containing minimization data.
"""
function minfile(h11,tri, cy=1)
    return string(geom_dir(h11,tri,cy),"/minima.h5")
end



end 