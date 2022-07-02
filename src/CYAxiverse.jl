module CYAxiverse



using Distributed

include("filestructure.jl")
include("read.jl")
include("generate.jl")
include("plotting.jl")

if ENV["PYTHON"] == "/opt/cytools/cytools-venv//bin/python3"
    include("../add_functions/cytools_wrapper.jl")
else
    println("This installation does not include CYTools!")
end

if haskey(ENV, "SLURM_JOB_ID")
    include("slurm.jl")
else
    println("This installation does not include SLURM!")
end

end
