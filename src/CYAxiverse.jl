module CYAxiverse

if haskey(ENV,"newARGS")
else
    println("Please specify where to read/write data, currently using pwd!")
end
include("filestructure.jl")
include("read.jl")
include("generate.jl")
include("plotting.jl")
include("../add_functions/profiling.jl")
if haskey(ENV, "PYTHON")
    if ENV["PYTHON"] == "/opt/cytools/cytools-venv//bin/python3"
        include("../add_functions/cytools_wrapper.jl")
    end
else
    println("This installation does not include CYTools!")
end

if haskey(ENV, "SLURM_JOB_ID")
    include("slurm.jl")
else
    println("This installation does not include SLURM!")
end

end
