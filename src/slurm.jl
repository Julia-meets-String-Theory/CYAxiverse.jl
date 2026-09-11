module slurm

"""Resolve the Slurm log directory from an argument or scheduler environment."""
function slurm_log_dir(log_dir::Union{Nothing,AbstractString}=nothing)
    selected = if log_dir !== nothing
        strip(String(log_dir))
    else
        configured = strip(get(ENV, "CYAXIVERSE_SLURM_LOG_DIR", ""))
        isempty(configured) ? strip(get(ENV, "SLURM_SUBMIT_DIR", "")) : configured
    end
    isempty(selected) && throw(ArgumentError(
        "Slurm log directory is not configured; pass log_dir or set " *
        "CYAXIVERSE_SLURM_LOG_DIR or SLURM_SUBMIT_DIR"))
    normpath(abspath(expanduser(selected)))
end

"""Append text to `slurm-<id>.out` in the configured Slurm log directory."""
function writeslurm(id::Union{Int,AbstractString}, s::AbstractString;
        log_dir::Union{Nothing,AbstractString}=nothing)
    directory = slurm_log_dir(log_dir)
    mkpath(directory)
    slurmlog = joinpath(directory, string("slurm-", id, ".out"))
    open(slurmlog, "a") do outf
        write(outf, s)
    end
    slurmlog
end

if haskey(ENV, "SLURM_ARRAY_TASK_ID")
    jobid = parse(Int64, ENV["SLURM_JOB_ID"])
    task_id = parse(Int64, ENV["SLURM_ARRAY_TASK_ID"])
    jobid = string(jobid, "_", task_id)
else
    jobid = parse(Int64, ENV["SLURM_JOB_ID"])
end

end
