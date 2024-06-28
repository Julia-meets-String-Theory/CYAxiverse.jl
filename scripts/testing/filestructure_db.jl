function np_path_generate(; geometric_data::Bool = false)
    np_paths = Vector{UInt8}[]
    np_pathinds = Vector{Int}[]
    for i in first(walkdir(CYAxiverse.filestructure.present_dir()))[2]
        if occursin(r"h11_*", i)
            for j in first(walkdir(joinpath(CYAxiverse.filestructure.present_dir(),i)))[2]
                if occursin(r"np_*", j)
                    if isfile(joinpath(CYAxiverse.filestructure.present_dir(),i,j,"K.hdf5"))
                        h11, tri, cy = parse(Int,SubString(i,5,7)), parse(Int,SubString(j,4,10)), 1
                        if geometric_data
                            if CYAxiverse.filestructure.isgeometry(h11, tri, cy)
                                push!(np_paths,transcode(UInt8,joinpath(i,j,"cy_0000001")))
                                push!(np_pathinds,[h11, tri, cy])
                            end
                        else
                            push!(np_paths,transcode(UInt8,joinpath(i,j,k)))
                            push!(np_pathinds,[h11, tri, cy])
                        end
                    end
                    for k in first(walkdir(joinpath(CYAxiverse.filestructure.present_dir(),i,j)))[2]
                        if occursin(r"cy_*", k)
                            if isfile(joinpath(CYAxiverse.filestructure.present_dir(),i,j,k,"K.hdf5"))
                                h11, tri, cy = parse(Int,SubString(i,5,7)), parse(Int,SubString(j,4,10)), parse(Int,SubString(k,4,10))
                                if geometric_data
                                    if CYAxiverse.filestructure.isgeometry(h11, tri, cy)
                                        push!(np_paths,transcode(UInt8,joinpath(i,j,k)))
                                        push!(np_pathinds,[h11, tri, cy])
                                    end
                                else
                                    push!(np_paths,transcode(UInt8,joinpath(i,j,k)))
                                    push!(np_pathinds,[h11, tri, cy])
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    hcat(np_paths...), hcat(np_pathinds...)
end