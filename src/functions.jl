#################
### Settings ####
#################
LinearAlgebra.BLAS.set_num_threads(1)
setprecision(ArbFloat,digits=5_000)

#######################
### Test functions ####
########################

function greet_CYAxiverse()
    println("Hello CYAxiverse")
end

###############################
### Initialising functions ####
###############################
"""
    localARGS()
Load key for data dir
"""
function localARGS()
    if @isdefined newARGS
        newARGS
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
        "vacua_test" => "/home/uni09/cosmo/mehta2/KSAxiverse_Jun20_InKC/KSAxiverse/vacua_test_data/", 
        "vacua_new" => "/scratch/users/mehta2/vacua_db/"
        )
    try
        ol_db[string(args)]
    catch y
        ol_db["home_Large"]
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
    mkpath(string(data_dir(),"logs"))
    log = string(Dates.DateTime(Dates.now()),"log.out")
    return joinpath(data_dir(), log)
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
Walks through data_dir() and returns list of data paths and matrix of [h11; tri; cy].
Saves in h5 file paths_cy.h5
"""
function np_path()
    np_paths = Vector{UInt8}[]
    np_pathinds = Vector{Int}[]
    nptest = 0
    for i in first(walkdir(present_dir()))[2]
        if occursin(r"h11_*", i)
            for j in first(walkdir(joinpath(present_dir(),i)))[2]
                if occursin(r"np_*", j)
                    for k in first(walkdir(joinpath(present_dir(),i,j)))[2]
                        if occursin(r"cy_*", k)
                            push!(np_paths,transcode(UInt8,joinpath(i,j,k)))
                            push!(np_pathinds,[parse(Int,SubString(i,5,7)),parse(Int,SubString(j,4,10)),parse(Int,SubString(k,4,10))])
#                                 println([parse(Int,SubString(i,5,7)),parse(Int,SubString(j,4,10)),parse(Int,SubString(k,4,10))])
                        end
                    end
                end
            end
        end
    end
    if isfile(joinpath(data_dir(),"paths.h5")) || isfile(joinpath(data_dir(),"paths_cy.h5"))
        h5open(joinpath(data_dir(),"paths_cy.h5"), "cw") do f
            f["paths",deflate=9] = hcat(np_paths...)
            f["pathinds",deflate=9] = hcat(np_pathinds...)
        end
    end
    return hcat(np_paths...), hcat(np_pathinds...)
end

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

function h11lst(h11min=0,h11max=100)
    pathinds_cy = paths_cy()[2]
    h11list = pathinds_cy[:,h11min .< pathinds_cy[1,:].<= h11max]
    return h11list
end

function pygmo_dir(h11,tri,cy=1)
    if localARGS()!=string("inKC")
        if localARGS()==string("home_Large")||localARGS()==string("KV1")
            if h11 >= 238
                if isdir(string(present_dir(),"h11_",lpad(h11,3,"0"),"/np_",lpad(tri,7,"0"),"/cy_",lpad(cy,7,"0"),"/pygmo"))
                    string(present_dir(),"h11_",lpad(h11,3,"0"),"/np_",lpad(tri,7,"0"),"/cy_",lpad(cy,7,"0"),"/pygmo")
                else
                    mkdir(string(present_dir(),"h11_",lpad(h11,3,"0"),"/np_",lpad(tri,7,"0"),"/cy_",lpad(cy,7,"0"),"/pygmo"))
                end
            else
                if isdir(string(present_dir(),"h11_",lpad(h11,3,"0"),"/np_",lpad(tri,7,"0"),"/pygmo"))
                    string(present_dir(),"h11_",lpad(h11,3,"0"),"/np_",lpad(tri,7,"0"),"/pygmo")
                else
                    mkdir(string(present_dir(),"h11_",lpad(h11,3,"0"),"/np_",lpad(tri,7,"0"),"/pygmo"))
                end
            end
        else
            if isdir(string(present_dir(),"h11_",lpad(h11,3,"0"),"/np_",lpad(tri,7,"0"),"/cy_",lpad(cy,7,"0"),"/pygmo"))
                    string(present_dir(),"h11_",lpad(h11,3,"0"),"/np_",lpad(tri,7,"0"),"/cy_",lpad(cy,7,"0"),"/pygmo")
            else
                mkdir(string(present_dir(),"h11_",lpad(h11,3,"0"),"/np_",lpad(tri,7,"0"),"/cy_",lpad(cy,7,"0"),"/pygmo"))
            end
        end
    else
        if isdir(string(present_dir(),"h11_",lpad(h11,3,"0"),"/np_",lpad(tri,7,"0"),"/pygmo"))
            string(present_dir(),"h11_",lpad(h11,3,"0"),"/np_",lpad(tri,7,"0"),"/pygmo")
        else
            mkdir(string(present_dir(),"h11_",lpad(h11,3,"0"),"/np_",lpad(tri,7,"0"),"/pygmo"))
        end
    end
end
#################
### Constant ####
#################

function constants()
    mplanck_r::ArbFloat = ArbFloat(2.435e18)
    hubble::ArbFloat = ArbFloat(2.13*0.7*1e-33)
    log2pi::ArbFloat = ArbFloat(log10(2*pi))
    return mplanck_r,hubble,log2pi
end

###################################
### Geometric Data Files (old) ####
###################################

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

function cyax_file(h11,tri, cy=1)
    return string(pygmo_dir(h11,tri,cy),"/cyax.h5")
end

function cyax_file(h11,tri, cy=1)
    return string(pygmo_dir(h11,tri,cy),"/cyax.h5")
end
