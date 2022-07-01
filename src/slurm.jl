module slurm

function writeslurm(id::Int,s::String)
    slurmlog = joinpath("/usr","users","mehta2","slurmlog",string("slurm-",id,".out"))
    open(slurmlog, "a") do outf
        write(outf,s)
    end
end
jobid = parse(Int64, ENV["SLURM_JOB_ID"])

end