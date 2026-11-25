# Phase 10: what does CYAxiverse.jl actually receive from a materialized G2 file?
# For every geometry under CYAXIVERSE_DATA_DIR, record (i) raw HDF5.jl shapes, (ii) the values returned by the
# readers the spectrum path uses (read.potential_factored -> pq_hybrid_physical_spectrum), and (iii) sampled
# charge columns, L columns and q_i' Kinv q_j contractions evaluated in Julia, for comparison with Python.
# Usage: CYAXIVERSE_DATA_DIR=<materialized root> julia --project=<CYAxiverse> julia_crosscheck.jl OUT_JSONL [max_files]
using CYAxiverse, HDF5, LinearAlgebra, Random

# minimal JSON writer (the CYAxiverse project environment is not modified to add a JSON package)
tojson(x::AbstractString) = "\"" * replace(x, "\\" => "\\\\", "\"" => "\\\"", "\n" => " ") * "\""
tojson(x::Bool) = x ? "true" : "false"
tojson(x::Integer) = string(x)
tojson(x::AbstractFloat) = isfinite(x) ? repr(Float64(x)) : "null"
tojson(x::Tuple) = tojson(collect(x))
tojson(x::AbstractVector) = "[" * join(tojson.(x), ",") * "]"
tojson(x::AbstractDict) = "{" * join([tojson(string(k)) * ":" * tojson(v) for (k, v) in x], ",") * "}"
tojson(x::Nothing) = "null"

out = ARGS[1]
maxfiles = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : typemax(Int)
root = ENV["CYAXIVERSE_DATA_DIR"]
files = String[]
for (dp, _, fns) in walkdir(root), fn in fns
    fn == "cyax.h5" && push!(files, joinpath(dp, fn))
end
sort!(files)
files = files[1:min(end, maxfiles)]
parseidx(p) = (m = match(r"h11_(\d+)/np_(\d+)/cy_(\d+)/cyax.h5$", p); parse.(Int, m.captures))

open(out, "w") do io
    for path in files
        h11, np_, cy = parseidx(path)
        g = CYAxiverse.structs.GeometryIndex(h11, np_, cy)
        rec = Dict{String,Any}("file" => relpath(path, root), "h11" => h11)
        try
            raw = h5open(path, "r") do f
                Dict("Q" => size(read(f["cytools/potential/Q"])), "L" => size(read(f["cytools/potential/L"])),
                     "Kinv" => size(read(f["cytools/geometric/Kinv"])),
                     "divisor_volumes" => size(read(f["cytools/geometric/divisor_volumes"])),
                     "effective_cone" => size(read(f["cytools/geometric/effective_cone"])),
                     "tip" => size(read(f["cytools/geometric/tip"])))
            end
            rec["julia_raw_shapes"] = Dict(k => collect(v) for (k, v) in raw)
            eff = h5open(f -> read(f["cytools/geometric/effective_cone"]), path, "r")   # Julia sees (h11, nq)
            geo = CYAxiverse.read.geometry(g)
            fac = CYAxiverse.read.potential_factored(g)
            Q, L, Kinv, C = fac.Q, fac.L, fac.Kinv, fac.C
            nq = size(eff, 2)
            rec["reader_Q_size"] = collect(size(Q)); rec["reader_L_size"] = collect(size(L))
            rec["reader_Kinv_size"] = collect(size(Kinv))
            rec["geometry_Kinv_equals_factored_Kinv"] = geo.kinv == Kinv
            rec["Kinv_raw_symmetric"] = geo.kinv == transpose(geo.kinv)
            rec["cholesky_residual"] = maximum(abs.(C * C' - Kinv)) / maximum(abs.(Kinv))
            rec["direct_columns_equal_effective_cone"] = Q[:, 1:nq] == eff
            rec["tip"] = geo.tip; rec["divisor_volumes"] = geo.τ_volumes; rec["CY_volume"] = geo.cy_volume
            rng = MersenneTwister(1000 * h11 + np_)
            N = size(Q, 2)
            cols = unique(vcat([1, N], rand(rng, 1:N, 8)))
            rec["sample_columns"] = cols
            rec["sample_Q"] = [Q[:, k] for k in cols]
            rec["sample_L"] = [L[:, k] for k in cols]
            pairs = [(rand(rng, 1:nq), rand(rng, 1:nq)) for _ in 1:10]
            rec["sample_pairs"] = [[i, j] for (i, j) in pairs]
            rec["sample_qKq"] = [dot(Q[:, i], Kinv * Q[:, j]) for (i, j) in pairs]
            o = CYAxiverse.read.oriented_potential(g)
            rec["oriented_Q_equals_raw"] = o.Q == Q
            rec["oriented_L_equals_raw"] = o.L == L
        catch e
            rec["error"] = first(split(sprint(showerror, e), "\n"))
        end
        println(io, tojson(rec)); flush(io)
    end
end
println("done ", length(files))
