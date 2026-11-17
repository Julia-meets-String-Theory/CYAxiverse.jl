# Phase 1 (+ structure for Phase 5): for each of the 46 geometries, recompute the theta = 0 production mass matrix
# (leading-instanton Hessian W at prec 500, exactly as pq_hybrid_physical_spectrum builds it), its full eigenvalue
# spectrum, and the exact structure of the leading-branch minimum:
#   leading signs, which Q columns carry the negative leading amplitudes (single vs pair term), |det Qtilde|,
#   and the EXACT rational phase q_a . theta* (mod 1) of every instanton at every leading-minimum lattice copy
#   theta* = Qtilde^{-T} (mask/2 + n), mask_i = 1 iff sign_i < 0.
# Read-only with respect to every input file.
# Usage: CYAXIVERSE_DATA_DIR=<WORK>/materialized julia --project=<code> 01_identify_theta0.jl LIST_TSV SPECTRA_REF OUT_JSONL
using CYAxiverse, HDF5, LinearAlgebra
const G = CYAxiverse.generate
include(joinpath(@__DIR__, "common.jl"))

rows = [split(l, '\t') for l in readlines(ARGS[1])[2:end]]
spectra = ARGS[2]; out = ARGS[3]; prec = 500
open(out, "w") do io
    for r in rows
        h, pos = parse(Int, r[1]), parse(Int, r[2])
        rec = Dict{String,Any}("h11" => h, "position" => pos, "geometry_id" => geom_id(h, pos))
        t0 = time()
        fac = CYAxiverse.read.potential_factored(CYAxiverse.structs.GeometryIndex(h, pos, 1))
        C, L, Q = fac.C, fac.L, fac.Q
        N = size(Q, 2); n_single = single_term_count(N)
        rec["n_instantons"] = N; rec["n_single_terms"] = n_single
        lq = G.LQtilde(Q, L); Lt, Qt = lq.Ltilde, lq.Qtilde
        cols = leading_columns(Q, L, Qt, Lt)
        neg = findall(<(0), Lt[1, :])
        rec["leading_columns"] = cols
        rec["negative_leading_columns"] = cols[neg]
        rec["negative_leading_is_pair_term"] = [c > n_single for c in cols[neg]]
        rec["negative_leading_log10_scale"] = Lt[2, neg]
        rec["n_negative_leading"] = length(neg)
        # theta = 0 production mass matrix and its complete eigenvalue spectrum
        W, _ = G.high_precision_leading_hessian(C, Lt, Qt; prec)
        T = eltype(W)
        lth = T(10)^(2 * (T(THR) - T(OFFSET)))
        M = Matrix(W)
        rec["inertia_above_H0"] = G.positive_inertia(bunchkaufman(Hermitian(M - lth * I)))
        rec["inertia_below_minus_H0"] = G.positive_inertia(bunchkaufman(Hermitian(-M - lth * I)))
        vals = eigvals(W)
        top = maximum(abs, vals)
        floorv = top * T(10)^(-(prec - 20))
        resolved = [v for v in vals if abs(v) > floorv]
        rec["eig_negative_resolved"] = count(<(0), resolved)
        rec["eig_negative_abs_m_ge_H0"] = count(v -> v < 0 && abs(v) >= lth, vals)
        rec["eig_positive_m_ge_H0"] = count(v -> v >= lth, vals)
        mostneg = minimum(vals)
        rec["most_negative_eigenvalue_log10_abs"] = Float64(log10(abs(mostneg)))
        rec["most_negative_abs_m_log10_eV"] = mass_log10(mostneg)
        rec["negative_abs_m_log10_eV"] = sort([mass_log10(v) for v in resolved if v < 0]; rev=true)
        rec["positive_m_ge_H0_log10_eV"] = sort([mass_log10(v) for v in vals if v >= lth]; rev=true)
        # stored spectrum
        m = h5open(spectra_file(spectra, h, pos), "r") do f
            read(f["spectrum/physical/m"])
        end
        rec["stored_physical_count"] = length(m)
        rec["stored_log10_m_eV"] = sort(m; rev=true)
        # exact leading-branch minimum structure
        ex = exact_leading_minima(Q, L; max_copies=4096)
        rec["det_Qtilde"] = ex.det
        rec["leading_minimum_copies_examined"] = length(ex.copies)
        rec["copies_all_phases_in_0_half"] = count(c -> c.all_half, ex.copies)
        rec["max_subleading_offgrid_log10_scale"] = maximum(c -> c.offgrid_max_log10, ex.copies)
        rec["n_offgrid_instantons_first_copy"] = ex.copies[1].n_offgrid
        rec["n_sign_flipped_instantons_first_copy"] = ex.copies[1].n_flipped
        rec["seconds"] = time() - t0
        println(io, tojson(rec)); flush(io)
        println(stderr, "done ", h, " ", pos, " ", round(time() - t0; digits=1), "s")
    end
end
println("done ", length(rows))
