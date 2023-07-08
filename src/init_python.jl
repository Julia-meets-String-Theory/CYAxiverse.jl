using Pkg

if haskey(ENV, "PYTHON")
    println("Your current PYTHON is $(ENV["PYTHON"])")
    if occursin("cytools", ENV["PYTHON"])
        try 
            using PyCall
            if PyCall.current_python() != ENV["PYTHON"]
                Pkg.build("PyCall")
                println("PyCall.jl has been built using $(ENV["PYTHON"])")
            end
        catch e
            Pkg.build("PyCall")
            println("PyCall.jl has been built using $(ENV["PYTHON"])")
        end
    else
        try 
            using PyCall
            if PyCall.current_python() != ENV["PYTHON"]
                Pkg.build("PyCall")
                println("PyCall.jl has been built using $(ENV["PYTHON"])")
            end
        catch e
            Pkg.build("PyCall")
            println("PyCall.jl has been built using $(ENV["PYTHON"])")
        end
    end
else
    error("Please set PYTHON with your PYTHONPATH")
end