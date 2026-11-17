# Read-only per-geometry accounting of axionic directions in the frozen thesis dataset.
# For each accepted geometry (unified view): rank(Q), LQtilde selection, sign of the leading amplitudes,
# PQ-frame (sequential, log-space) masses of ALL leading modes, arbitrary-precision leading-Hessian inertia
# (the production quantity) at the H0 threshold and at zero, and the stored physical spectrum.
# Nothing is written to the dataset.
# Usage: CYAXIVERSE_DATA_DIR=<WORK>/materialized julia --project=<code> 00_theta0_inertia_scan.jl LIST_TSV SPECTRA_VIEW OUT_JSONL [prec]
# LIST_TSV columns h11, position[, frst] (frst defaults to 1; the record carries "frst" only when the column is given).
using CYAxiverse, HDF5, LinearAlgebra
const G = CYAxiverse.generate
tojson(x::AbstractString) = "\"" * replace(x, "\\" => "\\\\", "\"" => "\\\"") * "\""
tojson(x::Bool) = x ? "true" : "false"
tojson(x::Integer) = string(x)
tojson(x::AbstractFloat) = isfinite(x) ? repr(Float64(x)) : (isnan(x) ? "null" : (x > 0 ? "1e999" : "-1e999"))
tojson(x::AbstractVector) = "[" * join(tojson.(x), ",") * "]"
tojson(x::AbstractDict) = "{" * join([tojson(string(k)) * ":" * tojson(v) for (k, v) in x], ",") * "}"
tojson(x::Nothing) = "null"

rows = [split(l, '\t') for l in readlines(ARGS[1])[2:end]]
spectra = ARGS[2]; out = ARGS[3]; prec = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 500
c = G.constants()
thr = Float64(log10(c["Hubble"]))
offset = 9 + Float64(log10(c["MPlanck"])) + Float64(c["log2π"])
open(out, "w") do io
    for r in rows
        h, pos = parse(Int, r[1]), parse(Int, r[2]); frst = length(r) >= 3 ? parse(Int, r[3]) : 1
        rec = Dict{String,Any}("h11" => h, "position" => pos)
        length(r) >= 3 && (rec["frst"] = frst)
        try
            t0 = time()
            g = CYAxiverse.structs.GeometryIndex(h, pos, frst)
            fac = CYAxiverse.read.potential_factored(g)
            C, L, Q = fac.C, fac.L, fac.Q
            rec["n_instantons"] = size(Q, 2)
            rec["rank_Q"] = rank(Float64.(Q))
            lq = G.LQtilde(Q, L)
            Lt, Qt = lq.Ltilde, lq.Qtilde
            rec["n_leading"] = size(Qt, 2)
            rec["n_leading_negative_sign"] = count(<(0), Lt[1, :])
            rec["leading_log10_scale_max"] = maximum(Lt[2, :]); rec["leading_log10_scale_min"] = minimum(Lt[2, :])
            _, mapprox, _ = G.pq_canonical_frame(Matrix(Qt') * C, Lt)
            pqm = mapprox .+ offset                                   # log10(m/eV), sequential PQ frame, one per leading instanton
            rec["pq_log10_m_eV"] = pqm
            rec["pq_sign"] = Int.(sign.(Lt[1, :]))
            rec["pq_above_H0"] = count(>=(thr), pqm)
            rec["pq_above_H0_positive_sign"] = count(i -> pqm[i] >= thr && Lt[1, i] > 0, eachindex(pqm))
            # log10 of the smallest resolvable |eigenvalue| ratio at `prec` digits: eigenvalue ~ 10^(2 m); dynamic range in dex
            top = maximum(pqm)
            rec["pq_eig_dynamic_range_dex"] = 2 * (top - minimum(pqm))
            rec["pq_modes_within_prec"] = count(m -> 2 * (top - m) <= prec - 20, pqm)
            W, _ = G.high_precision_leading_hessian(C, Lt, Qt; prec)
            T = eltype(W)
            lth = T(10)^(2 * (T(thr) - T(offset)))
            M = Matrix(W)
            rec["inertia_above_H0"] = G.positive_inertia(bunchkaufman(Hermitian(M - lth * I)))          # == production physical_count
            rec["inertia_below_minus_H0"] = G.positive_inertia(bunchkaufman(Hermitian(-M - lth * I)))   # tachyonic with |m| >= H0
            # unshifted inertia is a diagnostic only: sub-precision eigenvalues can round to exact zero pivots
            try
                rec["inertia_positive"] = G.positive_inertia(bunchkaufman(Hermitian(M)))
                rec["inertia_negative"] = G.positive_inertia(bunchkaufman(Hermitian(-M)))
            catch e
                e isa SingularException || rethrow()
                rec["W_numerically_singular_at_prec"] = true
            end
            cert = G._certified_float64_inertia_count(C, Lt, Qt, thr)
            rec["certified_float64_count"] = cert === nothing ? -1 : cert
            m = h5open(joinpath(spectra, "h11_" * lpad(h, 3, '0'), "np_" * lpad(pos, 7, '0'), "cy_" * lpad(frst, 7, '0'), "cyax.h5"), "r") do f
                read(f["spectrum/physical/m"])
            end
            rec["stored_physical_count"] = length(m)
            rec["stored_log10_m_eV"] = sort(m; rev=true)
            rec["seconds"] = time() - t0
        catch e
            rec["error"] = first(split(sprint(showerror, e), "\n"))
        end
        println(io, tojson(rec)); flush(io)
    end
end
println("done ", length(rows))
