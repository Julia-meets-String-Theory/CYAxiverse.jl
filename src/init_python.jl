using Pkg
if haskey(ENV, "PYTHON")
    if ENV["PYTHON"] == "/scratch/users/mehta2/cyaxiverse_python/bin/python"
    else
        ENV["PYTHON"] = "/scratch/users/mehta2/cyaxiverse_python/bin/python"
        Pkg.build("PyCall")
        
    end
end