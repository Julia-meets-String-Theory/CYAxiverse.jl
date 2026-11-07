# Phase 6: physical spectrum about the validated minimum theta_min, with the PRODUCTION engine and settings
# (generate.pq_hybrid_physical_spectrum internals, prec 500, quartics + mixed quartics, threshold log10(H0)) and the
# PRODUCTION writer (scripts/batch_physical_spectrum.jl::_write_result) into NEW files under corrected_spectra/.
#
# Re-expansion: V(theta_min + d) = sum_a s_a Lambda_a [1 - cos(2 pi q_a.d + phi_a)], phi_a = 2 pi q_a.theta_min.
# At a stationary point the sin(phi_a) pieces carry no quadratic or quartic terms, so the mass matrix and quartic
# couplings about theta_min are exactly those of the unphased potential with amplitudes s_a Lambda_a cos(phi_a):
# L_eff = (sign(s_a cos phi_a), L2_a + log10|cos phi_a|); instantons with cos phi_a = 0 contribute nothing and are
# omitted. For phases in {0, 1/2} this is a pure sign flip. Stored Q, L, Kinv are not modified.
#
# Also recomputes the theta = 0 spectrum in memory with the original L (must equal the frozen stored spectrum) and
# matches modes by canonical-eigenvector overlap against the complete theta = 0 leading-Hessian eigenbasis.
# Usage: CYAXIVERSE_DATA_DIR=<WORK>/materialized julia --project=<code> 04_vacuum_spectra.jl LIST_TSV SPECTRA_REF OUT_ROOT OUT_JSONL
using CYAxiverse, HDF5, LinearAlgebra, Logging, ArbNumerics
include(joinpath(dirname(Base.active_project()), "scripts", "batch_physical_spectrum.jl"))   # production writer, not run
include(joinpath(@__DIR__, "common.jl"))
const G = CYAxiverse.generate
const PREC = 500

function effective_L(L, phase)
    keep = Int[]; cols = Vector{Vector{Float64}}()
    for a in axes(L, 2)
        p = phase[a]
        c = p == 0 ? 1.0 : (p == 1 // 2 ? -1.0 : cospi(2 * BigFloat(p)))
        iszero(c) && continue
        push!(keep, a); push!(cols, [sign(L[1, a] * c), L[2, a] + Float64(log10(abs(c)))])
    end
    keep, reduce(hcat, cols)
end

function run_spectrum(C, L, Q, label)
    msgs = String[]
    sp = Logging.with_logger(_BatchWarningLogger(msgs)) do
        G._pq_hybrid_physical_spectrum_factored(C, L, Q; threshold_log10=THR, prec=PREC, quartics=true,
            mixed_quartics=true, label=label)
    end
    sp, any(occursin("provisional", lowercase(m)) for m in msgs), msgs
end

rows = [split(l, '\t') for l in readlines(ARGS[1])[2:end]]
spectra_ref, out_root, out_jsonl = ARGS[2], ARGS[3], ARGS[4]
rev = readchomp(`git -C $(dirname(Base.active_project())) rev-parse HEAD`)
open(out_jsonl, "w") do io
    for r in rows
        h, pos = parse(Int, r[1]), parse(Int, r[2])
        rec = Dict{String,Any}("h11" => h, "position" => pos, "geometry_id" => geom_id(h, pos))
        t0 = time()
        try
            g = CYAxiverse.structs.GeometryIndex(h, pos, 1)
            fac = CYAxiverse.read.potential_factored(g); Q, L, C = fac.Q, fac.L, fac.C
            Kpot = CYAxiverse.read.potential(g).K
            ex = exact_leading_minima(Q, L)
            chosen = something(findfirst(c -> c.all_half, ex.copies), 1)
            rec["chosen_copy"] = chosen; rec["n_copies"] = length(ex.copies)
            # stored (frozen) theta = 0 spectrum
            ref = spectra_file(spectra_ref, h, pos)
            old = h5open(ref, "r") do f
                p = f["spectrum/physical"]
                (; m=read(p["m"]), fpert=read(p["fpert_log10"]), ls=read(p["lambda_self_log10"]),
                   lss=read(p["lambda_self_sign"]), fK=read(p["fK_log10"]))
            end
            # theta = 0 recomputed in memory (reproducibility control, and eigenvectors for matching)
            sp0, prov0, _ = run_spectrum(C, L, Q, "theta0 h11=$h,polytope=$pos")
            rec["theta0_recomputed_equals_stored"] = length(sp0.m) == length(old.m) &&
                maximum(abs.(sp0.m .- old.m); init=0.0) < 1e-9 && maximum(abs.(sp0.λself .- old.ls); init=0.0) < 1e-9
            rec["theta0_recompute_max_abs_dm"] = maximum(abs.(sp0.m .- old.m); init=0.0)
            # complete theta = 0 leading-Hessian eigenbasis (canonical frame), resolved part only
            lq = ex.lq
            W0, _ = G.high_precision_leading_hessian(C, lq.Ltilde, lq.Qtilde; prec=PREC)
            E0 = eigen(W0)
            T0 = eltype(E0.values); lth0 = T0(10)^(2 * (T0(THR) - T0(OFFSET)))
            fl = maximum(abs, E0.values) * T0(10)^(-(PREC - 20))
            res0 = findall(v -> abs(v) > fl, E0.values)
            V0 = Float64.(E0.vectors[:, res0]); val0 = E0.values[res0]
            # spectra at every leading-minimum copy (mirror copies must agree); keep the chosen one
            allc = map(enumerate(ex.copies)) do (k, c)
                keep, Le = effective_L(L, c.phase)
                sp, prov, msgs = run_spectrum(C, Le, Q[:, keep], "thetamin h11=$h,polytope=$pos,copy=$k")
                (; sp, prov, msgs, keep, Le)
            end
            spc = allc[chosen]
            rec["copies_spectrum_max_abs_dm"] = maximum(x -> length(x.sp.m) == length(spc.sp.m) ?
                maximum(abs.(x.sp.m .- spc.sp.m); init=0.0) : Inf, allc)
            sp = spc.sp
            rec["provisional"] = spc.prov; rec["warnings"] = spc.msgs
            rec["n_instantons_used"] = length(spc.keep)
            # mode matching against the theta = 0 eigenbasis
            B = Float64.(sp.eigenvectors)
            ov = abs.(transpose(V0) * B)                      # (resolved theta0 modes) x (new physical modes)
            modes = Dict{String,Any}[]
            for j in axes(B, 2)
                i = argmax(ov[:, j]); v = val0[i]
                cls = v < 0 ? (abs(v) >= lth0 ? "former_tachyon_abs_m_ge_H0" : "former_tachyon_abs_m_lt_H0") :
                      (v >= lth0 ? "stored_at_theta0" : "was_below_H0_at_theta0")
                d = Dict{String,Any}("new_log10_m_eV" => sp.m[j], "new_fpert_log10" => sp.m[j] - 9 - 0.5 * sp.λself[j],
                    "new_lambda_self_sign" => sp.λselfsign[j], "new_lambda_self_log10" => sp.λself[j],
                    "theta0_eigen_sign" => Int(sign(v)), "theta0_abs_m_log10_eV" => mass_log10(v),
                    "overlap" => ov[i, j], "second_overlap" => length(res0) > 1 ? sort(ov[:, j]; rev=true)[2] : 0.0,
                    "class" => cls)
                if cls == "stored_at_theta0"
                    k = argmin(abs.(old.m .- mass_log10(v)))
                    d["old_log10_m_eV"] = old.m[k]; d["old_fpert_log10"] = old.fpert[k]
                    d["old_lambda_self_sign"] = old.lss[k]; d["old_lambda_self_log10"] = old.ls[k]
                    d["dlog10_m"] = sp.m[j] - old.m[k]; d["dlog10_fpert"] = d["new_fpert_log10"] - old.fpert[k]
                    d["lambda_self_sign_changed"] = sp.λselfsign[j] != old.lss[k]
                end
                push!(modes, d)
            end
            rec["modes"] = modes
            # old stored modes that are NOT above H0 any more
            matched_old = [d["old_log10_m_eV"] for d in modes if haskey(d, "old_log10_m_eV")]
            rec["old_modes_lost"] = [m for m in old.m if !any(x -> x == m, matched_old)]
            rec["old_count"] = length(old.m); rec["new_count"] = length(sp.m)
            rec["old_masses"] = old.m; rec["new_masses"] = sp.m
            # write the corrected spectrum (production writer) into a NEW copy of the geometry file
            dst = joinpath(out_root, geom_id(h, pos), "cyax.h5")
            mkpath(dirname(dst)); isfile(dst) && error("refusing to overwrite $dst")
            cp(ref, dst); chmod(dst, 0o644)
            _write_result(dst, g, sp; prec=PREC, threshold_log10=THR, quartics=true, runtime_seconds=time() - t0,
                provisional=spc.prov, fK=_fK_log10(Kpot),
                log_domain_diagnostics=G.instanton_scale_precision_diagnostics(spc.Le; linear_boundary_precision_digits=PREC))
            c = ex.copies[chosen]
            h5open(dst, "r+") do f
                md = f["spectrum/physical/metadata"]
                md["evaluation_point"] = "validated_leading_branch_minimum"
                md["vacuum_correction_script"] = "producers/vacuum/04_vacuum_spectra.jl"
                md["code_git_revision"] = rev
                vg = create_group(f, "vacuum_correction")
                vg["theta_min_numerator", deflate=9] = Int64.(numerator.(c.theta))
                vg["theta_min_denominator", deflate=9] = Int64.(denominator.(c.theta))
                vg["theta_min_units"] = "periods: axion field = 2 pi theta (theta in [0,1)^h11, stored-Q basis)"
                vg["construction"] = "theta_min = Qtilde^{-T}(mask/2 + n), mask_i = 1 iff leading amplitude sign < 0 (generate.leading_critical_branches, negative_mode_range=0:0)"
                vg["effective_amplitude_rule"] = "L_eff = (sign(s_a cos 2pi q_a.theta_min), L2_a + log10|cos 2pi q_a.theta_min|); cos = 0 terms omitted"
                vg["instantons_used", deflate=9] = Int64.(spc.keep)
                vg["leading_minimum_copy"] = chosen
                vg["leading_minimum_copies"] = length(ex.copies)
                vg["all_phases_in_0_half"] = Int(c.all_half)
                vg["theta0_spectrum_source"] = ref
            end
            rec["output"] = dst
        catch e
            rec["error"] = first(split(sprint(showerror, e), "\n"))
        end
        rec["seconds"] = time() - t0
        println(io, tojson(rec)); flush(io)
        println(stderr, "done ", h, " ", pos, " ", round(time() - t0; digits=1), "s ", get(rec, "error", ""))
    end
end
println("done ", length(rows))
