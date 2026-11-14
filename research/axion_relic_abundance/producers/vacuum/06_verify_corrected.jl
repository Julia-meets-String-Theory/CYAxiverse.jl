# Phase 9 end-to-end check of the 46 VACUUM_CORRECTED files, starting only from what is written in them:
#  - theta_min (vacuum_correction/theta_min_numerator/denominator) is a stationary point of the FULL potential
#    (all instantons) and has no resolved negative Hessian direction (inflation_points, ~500 digits);
#  - the stored spectrum/physical/m equals the eigenvalues >= lambda(H0) of the production mass matrix about theta_min,
#    obtained here by a DIFFERENT route (complete eigvals of W_eff, not the hybrid subspace solver);
#  - fK_log10 unchanged w.r.t. the frozen theta = 0 file; spectrum metadata threshold/prec as in production.
# Usage: CYAXIVERSE_DATA_DIR=<WORK>/materialized julia --project=<code> 06_verify_corrected.jl CORRECTED_TSV OUT_JSONL SPECTRA_REF
using CYAxiverse, HDF5, LinearAlgebra, ArbNumerics
const G = CYAxiverse.generate; const IP = CYAxiverse.inflation_points
include(joinpath(@__DIR__, "common.jl"))

# ARGS[1]: TSV (geometry_id, h11, position, spectrum_source) of the VACUUM_CORRECTED manifest records
recs = [split(l, '\t') for l in readlines(ARGS[1])[2:end]]
println(stderr, "corrected records: ", length(recs))
open(ARGS[2], "w") do io
    for m in recs
        h, pos, path = parse(Int, m[2]), parse(Int, m[3]), String(m[4])
        rec = Dict{String,Any}("geometry_id" => String(m[1]), "file" => path)
        g = CYAxiverse.structs.GeometryIndex(h, pos, 1)
        fac = CYAxiverse.read.potential_factored(g); Q, L, C = fac.Q, fac.L, fac.C
        K = Matrix(CYAxiverse.read.potential(g).K)
        num, den, mstored, fK, thrf, precf, evalpt = h5open(path, "r") do f
            read(f["vacuum_correction/theta_min_numerator"]), read(f["vacuum_correction/theta_min_denominator"]),
            read(f["spectrum/physical/m"]), read(f["spectrum/physical/fK_log10"]),
            read(f["spectrum/physical/metadata/threshold_log10"]), read(f["spectrum/physical/metadata/prec"]),
            read(f["spectrum/physical/metadata/evaluation_point"])
        end
        fK0 = h5open(f -> read(f["spectrum/physical/fK_log10"]), spectra_file(ARGS[3], h, pos))
        theta = Rational{BigInt}.(num) .// Rational{BigInt}.(den)
        phase = mod.(vec(transpose(Rational{BigInt}.(Q)) * theta), 1)
        ctx = IP.prepare_context(Q, L, K; precision_bits=1661)
        fl = BigFloat(10)^-480
        d = IP.diagnose(ctx, theta; zero_tolerance=fl)
        lth_W = BigFloat(10)^(2 * (BigFloat(THR) - BigFloat(OFFSET)))
        tol = lth_W * 4 * BigFloat(pi)^2 / BigFloat(10)^ctx.log_shift * BigFloat(10)^-30
        rec["gradient_ok"] = d.gradient_residual <= tol
        rec["full_hessian_no_resolved_negative"] = d.negative_modes == 0
        # production mass matrix about theta_min, complete eigenvalues (independent of the hybrid solver)
        keep = Int[]; cols = Vector{Vector{Float64}}()
        for a in axes(L, 2)
            p = phase[a]; c = p == 0 ? 1.0 : (p == 1 // 2 ? -1.0 : cospi(2 * BigFloat(p)))
            iszero(c) && continue
            push!(keep, a); push!(cols, [sign(L[1, a] * c), L[2, a] + Float64(log10(abs(c)))])
        end
        lq = G.LQtilde(Q[:, keep], reduce(hcat, cols))
        W, _ = G.high_precision_leading_hessian(C, lq.Ltilde, lq.Qtilde; prec=500)
        ev = eigvals(W); T = eltype(ev); lt = T(10)^(2 * (T(THR) - T(OFFSET)))
        mind = sort([mass_log10(v) for v in ev if v >= lt]; rev=true)
        rec["n_stored"] = length(mstored); rec["n_independent"] = length(mind)
        rec["masses_max_abs_diff"] = length(mind) == length(mstored) ? maximum(abs.(mind .- sort(mstored; rev=true)); init=0.0) : Inf
        rec["all_masses_ge_H0"] = all(>=(THR), mstored)
        rec["fK_unchanged"] = fK == fK0
        rec["metadata_ok"] = thrf == THR && precf == 500 && evalpt == "validated_leading_branch_minimum"
        rec["pass"] = rec["gradient_ok"] && rec["full_hessian_no_resolved_negative"] && rec["n_stored"] == rec["n_independent"] &&
                      rec["masses_max_abs_diff"] < 1e-8 && rec["all_masses_ge_H0"] && rec["fK_unchanged"] && rec["metadata_ok"]
        println(io, tojson(rec)); flush(io)
    end
end
println("done")
