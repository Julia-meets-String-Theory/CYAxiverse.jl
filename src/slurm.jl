module slurm

function writeslurm(id::Int,s::String)
    slurmlog = joinpath("/usr","users","mehta2","slurmlog",string("slurm-",id,".out"))
    open(slurmlog, "a") do outf
        write(outf,s)
    end
end

function writeslurm(id::String,s::String)
    slurmlog = joinpath("/usr","users","mehta2","slurmlog",string("slurm-",id,".out"))
    open(slurmlog, "a") do outf
        write(outf,s)
    end
end

if haskey(ENV, "SLURM_ARRAY_TASK_ID")
    jobid = parse(Int64, ENV["SLURM_JOB_ID"])
    task_id = parse(Int64, ENV["SLURM_ARRAY_TASK_ID"])
    jobid = string(jobid, "_", task_id)
else
    jobid = parse(Int64, ENV["SLURM_JOB_ID"])
end

end