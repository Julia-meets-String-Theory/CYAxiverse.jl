# Phase 4b + 5: candidate minima and their validation on the FULL potential (every instanton), per geometry.
#
# Candidates: the repository's leading-branch minima, i.e. generate.leading_critical_branches(LQtilde(Q,L);
# negative_mode_range=0:0) (branches whose leading selected potential has zero negative modes). They are
# reconstructed EXACTLY as theta* = Qtilde^{-T}(mask/2 + n) and cross-checked against the repository coordinates.
# This is the vacuum set the repository CLI (`batch_vacua_pipeline.jl --method auto`) counts as
# `square_reduced_potential_determinant_count` (N_vac = |det Qtilde|) without persisting coordinates.
#
# Validation (repository numerics, CYAxiverse.inflation_points, BigFloat):
#   value / gradient / Hessian of V = sum_a s_a 10^{L2_a} (1 - cos 2 pi q_a.theta) with ALL instantons,
#   generalized Hessian H v = m^2 K v (K = inverse of stored Kinv, exactly as read.potential builds it),
#   repository correct_stationary_point (damped Newton; must report :converged),
#   repository diagnose (gradient norms, inertia), repeated at ~500 and ~1000 decimal digits.
#   Exact analytic part: q_a.theta* mod 1 for every instanton (rational); if all lie in {0,1/2} the gradient is
#   identically zero; otherwise an explicit bound on the off-grid gradient is reported.
# Production-matrix check: the leading-instanton Hessian W (production mass matrix) of the potential re-expanded
#   about theta*, whose cosines carry the phases 2 pi q_a.theta*: amplitudes s_a cos(2 pi q_a.theta*).
# Read-only. Usage:
#   CYAXIVERSE_DATA_DIR=<WORK>/materialized julia --project=<code> 03_validate_minima.jl LIST_TSV OUT_JSONL
using CYAxiverse, HDF5, LinearAlgebra
const G = CYAxiverse.generate
const IP = CYAxiverse.inflation_points
include(joinpath(@__DIR__, "common.jl"))

const PREC_DIGITS = 500                        # production --prec
const BITS = (1661, 3322)                      # ~500 and ~1000 decimal digits
floor_rel(bits) = BigFloat(10)^(-(floor(Int, bits * log10(2)) - 20))   # relative eigenvalue floor of the linear arithmetic (BigFloat: no underflow)

"""Effective (sign, log10 scale) of every instanton after re-expanding about a point with exact phases."""
function effective_L(L, phase)
    keep = Int[]; Le = zeros(2, 0)
    cols = Vector{Vector{Float64}}()
    for a in axes(L, 2)
        p = phase[a]
        c = p == 0 ? 1.0 : (p == 1 // 2 ? -1.0 : cospi(2 * BigFloat(p)))
        iszero(c) && continue                   # cos = 0: no quadratic/quartic contribution at this point
        push!(keep, a)
        push!(cols, [sign(L[1, a] * c), L[2, a] + Float64(log10(abs(c)))])
    end
    keep, reduce(hcat, cols)
end

function ip_counts(ctx, theta, lth_W, bits)
    d = IP.diagnose(ctx, theta; zero_tolerance=floor_rel(bits))
    s = ctx.log_shift
    ev = d.hessian_eigenvalues                                       # H v = m^2 K v, log-shifted units, 2 pi periods
    conv = BigFloat(10)^s / (4 * BigFloat(pi)^2)                     # -> production W units
    evW = ev .* conv
    (; d, n_neg_resolved=d.negative_modes, n_zeroish=d.zeroish_modes, n_pos_resolved=d.positive_modes,
       n_neg_ge_H0=count(v -> v < 0 && abs(v) >= lth_W, evW), n_pos_ge_H0=count(v -> v >= lth_W, evW),
       min_eig_sign=Int(sign(minimum(evW))), min_eig_abs_mass_log10=mass_log10(minimum(evW)),
       floor_mass_log10=mass_log10(maximum(abs, evW) * floor_rel(bits)),
       masses_ge_H0=sort([mass_log10(v) for v in evW if v >= lth_W]; rev=true),
       neg_masses=sort([mass_log10(v) for v in evW if v < -abs(maximum(abs, evW)) * floor_rel(bits)]; rev=true))
end

rows = [split(l, '\t') for l in readlines(ARGS[1])[2:end]]
open(ARGS[2], "w") do io
    for r in rows
        h, pos = parse(Int, r[1]), parse(Int, r[2])
        rec = Dict{String,Any}("h11" => h, "position" => pos, "geometry_id" => geom_id(h, pos))
        t0 = time()
        try
            g = CYAxiverse.structs.GeometryIndex(h, pos, 1)
            fac = CYAxiverse.read.potential_factored(g)
            Q, L, C = fac.Q, fac.L, fac.C
            K = Matrix(CYAxiverse.read.potential(g).K)
            op = CYAxiverse.read.oriented_potential(g)
            rec["oriented_potential_identical_to_production_inputs"] = op.Q == Q && op.L == L
            ex = exact_leading_minima(Q, L)
            lq = ex.lq
            rec["det_Qtilde"] = ex.det
            br = G.leading_critical_branches(lq; negative_mode_range=0:0)
            rec["repository_leading_minima_count"] = br.leading_minima_count
            rec["repository_branch_count"] = br.branch_count
            dist = [minimum(j -> maximum(min.(abs.(Float64.(c.theta) .- br.coordinates[:, j]),
                                              1 .- abs.(Float64.(c.theta) .- br.coordinates[:, j]))), axes(br.coordinates, 2))
                    for c in ex.copies]
            rec["exact_vs_repository_theta_max_torus_distance"] = maximum(dist)
            lth_W = BigFloat(10)^(2 * (BigFloat(THR) - BigFloat(OFFSET)))
            ctx = IP.prepare_context(Q, L, K; precision_bits=BITS[1])
            lth_ip = lth_W * 4 * BigFloat(pi)^2 / BigFloat(10)^ctx.log_shift
            tol_grad = lth_ip * BigFloat(10)^-30          # displacement <= 1e-30 (periods) along every mode with m >= H0
            rec["gradient_tolerance_log10"] = Float64(log10(tol_grad))
            rec["log_shift"] = ctx.log_shift
            # theta = 0 on the full potential (reference)
            z = ip_counts(ctx, zeros(Rational{BigInt}, h), lth_W, BITS[1])
            rec["theta0_full_neg_ge_H0"] = z.n_neg_ge_H0; rec["theta0_full_neg_resolved"] = z.n_neg_resolved
            rec["theta0_full_pos_ge_H0"] = z.n_pos_ge_H0
            rec["theta0_gradient_residual_log10"] = Float64(log10(max(z.d.gradient_residual, floatmin(Float64))))
            copies = Dict{String,Any}[]
            for (k, c) in enumerate(ex.copies)
                cr = Dict{String,Any}("copy" => k, "all_phases_in_0_half" => c.all_half, "n_offgrid" => c.n_offgrid,
                                      "theta_numerator" => [numerator(x) for x in c.theta],
                                      "theta_denominator" => [denominator(x) for x in c.theta])
                # analytic gradient bound from off-grid phases (log-shifted units), exact zero otherwise
                off = findall(p -> !(p == 0 || p == 1 // 2), c.phase)
                gb = isempty(off) ? BigFloat(0) :
                     2 * BigFloat(pi) * sum(BigFloat(10)^(BigFloat(L[2, b]) - ctx.log_shift) * maximum(abs, Q[:, b]) for b in off)
                cr["analytic_gradient_bound_log10"] = iszero(gb) ? -Inf : Float64(log10(gb))
                cr["offgrid_max_log10_scale"] = c.offgrid_max_log10
                # repository stationary-point correction from the exact seed
                corr = IP.correct_stationary_point(ctx, c.theta; residual_tolerance=tol_grad, max_iterations=100)
                cr["correction_status"] = string(corr.status)
                cr["correction_iterations"] = corr.iterations
                cr["correction_residual_log10"] = Float64(log10(max(corr.residual, BigFloat(10)^-100000)))
                cr["correction_moved_torus_distance_log10"] = let dth = abs.(corr.theta .- BigFloat.(c.theta))
                    m = maximum(min.(dth, 1 .- dth)); iszero(m) ? -Inf : Float64(log10(m)) end
                for (bi, bits) in enumerate(BITS)
                    cx = bi == 1 ? ctx : IP.prepare_context(Q, L, K; precision_bits=bits)
                    a = ip_counts(cx, c.theta, lth_W, bits)
                    tag = "p" * string(bits)
                    cr[tag * "_gradient_residual_inf_log10"] = Float64(log10(max(a.d.gradient_residual, BigFloat(10)^-100000)))
                    cr[tag * "_gradient_metric_norm_log10"] = Float64(log10(max(a.d.gradient_norm, BigFloat(10)^-100000)))
                    cr[tag * "_neg_resolved"] = a.n_neg_resolved
                    cr[tag * "_zeroish_below_floor"] = a.n_zeroish
                    cr[tag * "_pos_resolved"] = a.n_pos_resolved
                    cr[tag * "_neg_ge_H0"] = a.n_neg_ge_H0
                    cr[tag * "_pos_ge_H0"] = a.n_pos_ge_H0
                    cr[tag * "_min_eig_sign"] = a.min_eig_sign
                    cr[tag * "_min_eig_abs_mass_log10"] = a.min_eig_abs_mass_log10
                    cr[tag * "_noise_floor_mass_log10"] = a.floor_mass_log10
                    cr[tag * "_masses_ge_H0"] = a.masses_ge_H0
                    cr[tag * "_neg_masses_resolved"] = a.neg_masses
                    if bi == 1
                        V = a.d.value * BigFloat(10)^ctx.log_shift                  # V(theta*) - V(0), V(0) = 0 exactly
                        cr["V_min_sign"] = Int(sign(V)); cr["V_min_log10_abs"] = Float64(log10(abs(V)))
                        cr["V0"] = 0.0
                    end
                end
                # production mass matrix of the re-expanded potential (prec 500): inertia at +-lambda(H0)
                keep, Le = effective_L(L, c.phase)
                lqe = G.LQtilde(Q[:, keep], Le)
                W, _ = G.high_precision_leading_hessian(C, lqe.Ltilde, lqe.Qtilde; prec=PREC_DIGITS)
                T = eltype(W); lt = T(10)^(2 * (T(THR) - T(OFFSET))); M = Matrix(W)
                cr["W_eff_inertia_above_H0"] = G.positive_inertia(bunchkaufman(Hermitian(M - lt * I)))
                cr["W_eff_inertia_below_minus_H0"] = G.positive_inertia(bunchkaufman(Hermitian(-M - lt * I)))
                cr["W_eff_leading_negative_signs"] = count(<(0), lqe.Ltilde[1, :])
                evW = eigvals(W)
                cr["W_eff_masses_ge_H0"] = sort([mass_log10(v) for v in evW if v >= lt]; rev=true)
                cr["W_eff_resolved_negative"] = count(v -> v < -maximum(abs, evW) * T(10)^(-(PREC_DIGITS - 20)), evW)
                cr["W_eff_leading_selection_unchanged"] = sort(keep[leading_columns(Q[:, keep], Le, lqe.Qtilde, lqe.Ltilde)]) ==
                                                          sort(leading_columns(Q, L, lq.Qtilde, lq.Ltilde))
                cr["n_instantons_zero_amplitude_at_point"] = size(Q, 2) - length(keep)
                push!(copies, cr)
            end
            rec["copies"] = copies
        catch e
            rec["error"] = first(split(sprint(showerror, e), "\n"))
        end
        rec["seconds"] = time() - t0
        println(io, tojson(rec)); flush(io)
        println(stderr, "done ", h, " ", pos, " ", round(time() - t0; digits=1), "s ", haskey(rec, "error") ? rec["error"] : "")
    end
end
println("done ", length(rows))
