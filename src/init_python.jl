using Pkg
if haskey(ENV, "PYTHON")
    if ENV["PYTHON"] == "/home/cytools/cytools-venv//bin/python"
        using PyCall
        if PyCall.current_python() != ENV["PYTHON"]
            Pkg.build("PyCall")
        end
    elseif ENV["PYTHON"] == "/scratch/users/mehta2/cyaxiverse_python/bin/python"
        using PyCall
        if PyCall.current_python() != ENV["PYTHON"]
            Pkg.build("PyCall")
        end
    else
        error("Please set ENV['PYTHON'] with your PYTHONPATH")
    end
end