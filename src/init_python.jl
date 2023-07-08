using Pkg

if haskey(ENV, "PYTHON")
    println("Your current PYTHON is $(ENV["PYTHON"])")
    if ENV["PYTHON"] == "/home/cytools/cytools-venv//bin/python3"
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
    elseif ENV["PYTHON"] == "/scratch/users/mehta2/cyaxiverse_python/bin/python"
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