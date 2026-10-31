# Read back the vacua_pipeline groups written by the repository CLI into the WRITABLE COPIES (Phase 4a).
using HDF5
include(joinpath(@__DIR__, "common.jl"))
root = ARGS[1]; rows = [split(l, '\t') for l in readlines(ARGS[2])[2:end]]
open(ARGS[3], "w") do io
    for r in rows
        h, pos = parse(Int, r[1]), parse(Int, r[2])
        rec = Dict{String,Any}("h11" => h, "position" => pos)
        h5open(joinpath(root, geom_id(h, pos), "cyax.h5"), "r") do f
            g = f["vacua_pipeline"]
            rec["datasets"] = keys(g)
            for k in ("estimate", "issquare", "verified", "threshold")
                haskey(g, k) && (rec[k] = read(g[k]))
            end
            md = g["metadata"]
            for k in ("auto_selected_method", "search_method", "search_classification", "verification_status",
                      "model_scope", "full_potential_status", "det_Qtilde", "branch_count", "critical_count",
                      "minimum_count", "search_status", "status")
                haskey(md, k) && (rec[k] = read(md[k]))
            end
            rec["has_theta_min"] = haskey(g, "theta_min")
        end
        println(io, tojson(rec))
    end
end
