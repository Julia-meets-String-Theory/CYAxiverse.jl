# FINAL theta analysis: exact closed TIGHT intervals [theta_90, theta_100] on the domain 0 < theta <= theta_max (default 0.5;
# NO lower cutoff), every T_reh scenario of the relic run from one code path. Column names ending in 0p5 refer to theta_max. Grids are used only to tabulate curves for plotting; every reported maximum, interval
# and count at a stated angle is exact.
# Populations (approved 2026-10-31; abundance validity from relic/tables/geometry_abundance_validity_<label>.csv):
#   PRIMARY (theta/)                       EVALUATED_POSITIVE + EVALUATED_ZERO: complete abundances (0 for an evaluated empty sum)
#   SENSITIVITY_INCLUDING_PARTIAL_VALIDITY (theta_sensitivity_partial_validity/): PRIMARY + PARTIAL_VALIDITY with their partial
#                                          sums over valid modes (not complete abundances); explicitly labelled
#   INFLATION_INVALID_ALL geometries are in neither (abundance not evaluated); every geometry is listed in theta/theta_population.csv
include(joinpath(@__DIR__, "common.jl"))
const OPT = CFG["optimisation"]
const HS = Float64.(OPT["smoothing_half_widths_dex"]); const PF = Float64(OPT["plateau_fraction"])
const NB = Int(OPT["bootstrap_resamples"]); const SEED = Int(OPT["bootstrap_seed"])
const PH = Float64(get(OPT, "primary_smoothing_half_width_dex", 0.1))   # smoothing of the plateau and robustness rules
const RC = Dict{String,Any}("max_smoothed_max_shift_dex" => 0.3, "max_bootstrap_p16_p84_dex" => 0.5, "max_plateau80_width_dex" => 1.0)
merge!(RC, get(OPT, "robust_criteria", Dict{String,Any}()))
const LTH_MAX = log10(TH_MAX)

tiedrank(v) = (p = sortperm(v); r = similar(v, Float64); i = 1; n = length(v);
    while i <= n; j = i; while j < n && v[p[j+1]] == v[p[i]]; j += 1; end; r[p[i:j]] .= (i + j) / 2; i = j + 1; end; r)
spearman(x, y) = cor(tiedrank(x), tiedrank(y))
slope(x, y) = cov(x, y) / var(x)

"""Exact profile + maximum on (0, hi] for TIGHT band [xlo, X_HI]."""
function prof(om; xlo=X_LO, hi=TH_MAX)
    a = theta_at(om, xlo); b = theta_at(om, X_HI)
    P, vpt, vseg = exact_profile(a, b, 0.0, hi); M, ivs = maximal_set(P, vpt, vseg)
    M == 0 && (ivs = Vector{Vector{Float64}}())      # NO_MATCH: no geometry in the band anywhere on (0, hi]: no maximising set, no angle
    (; a, b, P, vpt, vseg, M, ivs, rep=M == 0 ? nothing : representative(ivs))
end
"""Upper-boundary rule of the approved theta analysis (unchanged), with no lower edge."""
function upper_status(om, A; xlo=X_LO)
    U = prof(om; xlo=xlo, hi=pi)
    tt = exp10.(range(log10(TH_MAX / 2), LTH_MAX; length=401)); Nt = evaluate_step(A.P, A.vpt, A.vseg, tt)
    sl = slope(log10.(tt), Float64.(Nt)); mhi = mean(Nt)
    mlo = mean(evaluate_step(A.P, A.vpt, A.vseg, exp10.(range(log10(TH_MAX / 4), log10(TH_MAX / 2); length=401))))
    touches = any(iv -> iv[2] == TH_MAX, A.ivs); rising = sl > 0 && mhi > mlo
    unc_beyond = U.M > A.M && minimum(iv[1] for iv in U.ivs) > TH_MAX
    bl = touches || (rising && unc_beyond)
    return (status=A.M == 0 ? "NO_MATCH" : bl ? "BOUNDARY_LIMITED" : "INTERIOR", touches_upper=touches, rising_into_upper=rising, slope_top_octave=sl,
            mean_top_octave=mhi, mean_second_octave=mlo, unconstrained_to_pi_N_max=U.M, unconstrained_to_pi_rep=U.rep, unconstrained_max_above_0p5=unc_beyond)
end
"""Smoothed count with the window truncated at the upper domain edge (no lower edge exists)."""
function smoothed_domain(l90s, L; h=PH)
    out = similar(collect(Float64, L))
    for (k, l) in enumerate(L)
        lo, hi = l - h, min(l + h, LTH_MAX)
        i1 = searchsortedfirst(l90s, lo - W); i2 = searchsortedlast(l90s, hi)
        s = 0.0
        @inbounds for j in i1:i2
            x = l90s[j]; s += max(0.0, min(x + W, hi) - max(x, lo))
        end
        out[k] = s / (hi - lo)
    end
    out
end
function plateau(L, s, frac)
    M = maximum(s); idx = findall(s .>= frac * M)
    (from=exp10(L[first(idx)]), to=exp10(L[last(idx)]), width_dex=L[last(idx)] - L[first(idx)], contiguous=(last(idx) - first(idx) + 1 == length(idx)))
end

const Gin = inputs_csv(); const inmap = Dict(r.geometry_id => r for r in eachrow(Gin))
hbin(h) = h <= 3 ? "01-03" : h <= 10 ? "04-10" : h <= 20 ? "11-20" : h <= 30 ? "21-30" : h <= 40 ? "31-40" : h <= 50 ? "41-50" : "51+"
function analyse(OUT, POP, sel)
geo_rows = DataFrame(); summ = Dict{String,Any}(); sum_rows = DataFrame(); iv_rows = DataFrame(); h11_rows = DataFrame(); hb_rows = DataFrame()
q_rows = DataFrame(); boot_rows = DataFrame(); nsize = Dict{String,Int}()
global_min90 = Inf
for lab in LABS
    G = geometry(lab); Mt = master(lab); V = validity(lab)
    V.geometry_id == G.geometry_id || error("$lab: validity table rows differ from the geometry table")
    keep = sel.(V.theta_population)
    # abundance used: the complete abundance (PRIMARY) or the partial sum over valid modes (PARTIAL_VALIDITY, sensitivity only)
    omv = [p == "PRIMARY" ? c : q for (p, c, q) in zip(V.theta_population, V.Omega_complete_h2_at_theta_ref, V.Omega_valid_modes_partial_h2_at_theta_ref)]
    G = G[keep, :]; omv = omv[keep]; nsize[lab] = nrow(G)
    if nrow(G) == 0
        summ[lab] = Dict("T_reh_GeV" => TREH[lab], "status" => "EMPTY_POPULATION", "n_geometries" => 0)
        continue
    end
    nmiss = count(ismissing, omv)
    nmiss == 0 || error("$lab: $nmiss geometries of population $POP have no abundance value")
    om = Float64.(omv)
    all(isfinite, om) && all(om .>= 0) || error("non-finite or negative Omega in $lab")
    # a geometry with Omega = 0 (no CDM-eligible mode) never reaches the band: theta thresholds +Inf, underabundant at every angle;
    # it stays in every count and fraction of the population
    A = prof(om); st = upper_status(om, A); NM = A.M == 0
    t10 = theta_at(om, 0.1); t80 = theta_at(om, 0.8); t95 = theta_at(om, 0.95); t90 = A.a; t100 = A.b
    global_min90 = min(global_min90, minimum(t90))
    cls = NM ? Vector{Union{Missing,String}}(missing, length(om)) : classify_at(A.rep, t90, t100)
    Rrep = NM ? Vector{Union{Missing,Float64}}(missing, length(om)) : om .* (A.rep / TH_REF)^2 ./ ODM
    pre = coalesce.(G.Omega_pre_reheating_h2_at_theta_ref, 0.0); post = coalesce.(G.Omega_post_reheating_h2_at_theta_ref, 0.0)
    all(o == 0 ? p + q == 0 : abs((p + q) / o - 1) < 1e-12 for (p, q, o) in zip(pre, post, om)) || error("PRE+POST != total in $lab")
    pdom = [o > 0 && p / o > 0.5 for (p, o) in zip(pre, om)]; qdom = [o > 0 && p / o <= 0.5 for (p, o) in zip(pre, om)]   # PRE-/POST-dominated (Omega > 0)
    share(x) = [o > 0 ? y / o : missing for (y, o) in zip(x, om)]
    # dominant mode attributes from the master table
    mk = Dict((r.geometry_id, r.mode_index) => r for r in eachrow(Mt))
    dom = [ismissing(i) ? missing : mk[(g, Int(i))] for (g, i) in zip(G.geometry_id, G.dominant_mode_index)]   # no dominant mode when Omega = 0
    n = nrow(G)
    df = DataFrame(geometry_id=G.geometry_id, h11=G.h11, tranche=[inmap[g].tranche for g in G.geometry_id], position=[inmap[g].position for g in G.geometry_id],
        vacuum_status=G.vacuum_status, qcd_edge_flag=[inmap[g].qcd_edge_flag for g in G.geometry_id], scenario=lab, T_reh_GeV=G.T_reh_GeV,
        theta_ref=G.theta_ref, Omega_geometry_h2_at_theta_ref=om, Omega_pre_reheating_h2_at_theta_ref=pre, Omega_post_reheating_h2_at_theta_ref=post,
        pre_share=share(pre), post_share=share(post), R_at_theta_ref=om ./ ODM,
        theta_10=t10, theta_80=t80, theta_90=t90, theta_95=t95, theta_100=t100,
        theta_90_le_0p5=t90 .<= TH_MAX, theta_100_le_0p5=t100 .<= TH_MAX,
        theta_rep_scenario=something(A.rep, missing), R_at_theta_rep=Rrep, class_at_theta_rep=cls,
        number_of_modes=G.number_of_modes, number_frozen=G.number_frozen, number_late=G.number_late, number_post_reheating=G.number_post_reheating,
        number_pre_reheating=G.number_pre_reheating, number_pre_reheating_valid=G.number_pre_reheating_valid, number_inflation_excluded=G.number_inflation_excluded,
        number_included=G.number_included, number_above_mKK=G.number_above_mKK, number_above_fpert=G.number_above_fpert,
        number_included_above_mKK=G.number_included_above_mKK, number_included_above_fpert=G.number_included_above_fpert,
        dominant_mode_index=G.dominant_mode_index, dominant_fraction=G.dominant_fraction, dominant_category=[ismissing(d) ? missing : d.category for d in dom],
        dominant_log10_mass_eV=[ismissing(d) ? missing : d.log10_mass_eV for d in dom], dominant_log10_f_pert_GeV=[ismissing(d) ? missing : d.log10_f_pert_GeV for d in dom],
        cy_volume=G.cy_volume)
    append!(geo_rows, df)

    # ---- distributions A, B ----
    for (nm, v) in (("theta_90", t90), ("theta_100", t100), ("theta_90_PRE_dominated", t90[pdom]), ("theta_90_POST_dominated", t90[qdom]))
        isempty(v) && continue                 # e.g. no PRE-dominated geometry in a small sample
        push!(q_rows, (scenario=lab, quantity=nm, n=length(v), min=minimum(v), p01=qtl(v, 0.001), p1=qtl(v, 0.01), p5=qtl(v, 0.05), p16=qtl(v, 0.16), p25=qtl(v, 0.25),
                       p50=qtl(v, 0.5), p75=qtl(v, 0.75), p84=qtl(v, 0.84), p95=qtl(v, 0.95), max=maximum(v),
                       n_le_0p5=count(v .<= TH_MAX), n_gt_0p5=count(v .> TH_MAX), n_gt_1=count(v .> 1), n_gt_pi=count(v .> pi)); promote=true)
    end
    # ---- exact step, maximal intervals ----
    csvw(joinpath(OUT, "curves", "theta_exact_step_$(lab).csv"), DataFrame(theta=A.P, N_tight_at_point=A.vpt, N_tight_on_following_open_segment=vcat(A.vseg, -1)))
    for (k, iv) in enumerate(A.ivs)
        push!(iv_rows, (scenario=lab, tight_lower=X_LO, role="PRIMARY", interval=k, theta_from=iv[1], theta_to=iv[2], N_tight=A.M, width_dex=log10(iv[2] / iv[1])))
    end
    # ---- smoothing (exact overlap formula; grid only tabulates the curve) ----
    m90 = any(isfinite, t90) ? minimum(t90[isfinite.(t90)]) : TH_MAX     # Omega = 0 geometries (theta_90 = Inf) add nothing to the smoothed count
    l90s = sort(log10.(t90)); L = collect(range(min(round(log10(m90) - 0.5; digits=1), round(LTH_MAX - 1.0; digits=1)), LTH_MAX; step=0.001))
    sm = Dict(h => smoothed_domain(l90s, L; h=h) for h in HS)
    csvw(joinpath(OUT, "curves", "theta_smoothed_count_$(lab).csv"),
         DataFrame(:theta => exp10.(L), [Symbol("N_tight_smoothed_pm$(replace(string(h), "." => "p"))dex") => sm[h] for h in HS]...))
    # NO_MATCH: the smoothed count is 0 everywhere; no maximum location, plateau or excess is defined
    smax = Dict(h => (theta=NM ? nothing : exp10(L[argmax(sm[h])]), value=maximum(sm[h])) for h in HS)
    shift = NM ? nothing : maximum(log10(smax[h].theta) for h in HS) - minimum(log10(smax[h].theta) for h in HS)
    p80 = NM ? nothing : plateau(L, sm[PH], PF); p50 = NM ? nothing : plateau(L, sm[PH], 0.5)
    s_rep = NM ? nothing : smoothed_domain(l90s, [log10(A.rep)]; h=PH)[1]
    # ---- exact class fractions on a plotting grid (each point exact) ----
    Lg = collect(range(-10.0, LTH_MAX; step=0.005)); s90 = sort(t90); s100 = sort(t100); s10 = sort(t10)
    csvw(joinpath(OUT, "curves", "theta_class_counts_vs_theta_$(lab).csv"), DataFrame(theta=exp10.(Lg),
        N_tight=[searchsortedlast(s90, x) - (searchsortedfirst(s100, x) - 1) for x in exp10.(Lg)],
        N_overabundant=[searchsortedfirst(s100, x) - 1 for x in exp10.(Lg)], N_underabundant=[n - searchsortedlast(s90, x) for x in exp10.(Lg)],
        N_strongly_underabundant=[n - searchsortedlast(s10, x) for x in exp10.(Lg)]))
    # ---- tight lower-edge sensitivity (diagnostic) ----
    edge = Dict{String,Any}()
    for xl in Float64.(OPT["diagnostic_tight_lower_edges"])
        D = prof(om; xlo=xl); sd = upper_status(om, D; xlo=xl)
        for (k, iv) in enumerate(D.ivs)
            push!(iv_rows, (scenario=lab, tight_lower=xl, role="DIAGNOSTIC", interval=k, theta_from=iv[1], theta_to=iv[2], N_tight=D.M, width_dex=log10(iv[2] / iv[1])))
        end
        edge[string(xl)] = Dict("N_max" => D.M, "theta_rep" => D.rep, "status" => sd.status, "n_intervals" => length(D.ivs))
    end
    # ---- bootstrap (geometry resampling) ----
    rng = MersenneTwister(SEED)
    for b in 1:NB
        idx = rand(rng, 1:n, n); Bp = prof(om[idx]); bs = upper_status(om[idx], Bp)
        push!(boot_rows, (scenario=lab, resample=b, N_max=Bp.M, theta_rep=something(Bp.rep, missing), status=bs.status); promote=true)
    end
    bb = boot_rows[boot_rows.scenario .== lab, :]; lb = log10.(collect(Float64, skipmissing(bb.theta_rep)))   # resamples without match have no angle
    bq = (p16=qtl(lb, 0.16), p50=qtl(lb, 0.5), p84=qtl(lb, 0.84))
    robust = !NM && st.status == "INTERIOR" && shift <= RC["max_smoothed_max_shift_dex"] && (bq.p84 - bq.p16) <= RC["max_bootstrap_p16_p84_dex"] && p80.width_dex <= RC["max_plateau80_width_dex"]
    descriptor = NM ? "NO_MATCH: no geometry reaches the TIGHT band $(X_LO) <= R <= $(X_HI) anywhere on 0 < theta <= $(TH_MAX); no representative angle" :
                 st.status == "BOUNDARY_LIMITED" ? "BOUNDARY_LIMITED: no interior preferred angle (count still rising at theta_max = $(TH_MAX))" :
                 robust ? "ROBUST_INTERIOR_ANGLE" : @sprintf("BROAD_INTERIOR_SCALE (no robust preferred angle): smoothed count >= 80%% of its maximum on %.2g-%.2g (%.1f dex)", p80.from, p80.to, p80.width_dex)
    c = NM ? nothing : Dict(k => count(isequal(k), cls) for k in ("TIGHT_DM", "ALLOWED_UNDERABUNDANT", "OVERABUNDANT"))
    nstrong = NM ? nothing : count(A.rep .< t10)
    nzero = count(==(0.0), om)
    rho = spearman(Float64.(G.h11), log10.(t90))
    summ[lab] = Dict("T_reh_GeV" => TREH[lab], "N_tight_max" => A.M, "f_tight_max" => A.M / n, "maximal_intervals" => A.ivs, "theta_rep" => A.rep,
        "upper_boundary" => Dict(string(k) => v for (k, v) in pairs(st)), "robust_preferred_angle" => robust, "descriptor" => descriptor,
        "counts_at_theta_rep" => NM ? nothing : Dict("TIGHT_DM" => c["TIGHT_DM"], "ALLOWED_UNDERABUNDANT" => c["ALLOWED_UNDERABUNDANT"], "OVERABUNDANT" => c["OVERABUNDANT"],
                                      "strongly_underabundant_R_lt_0p1" => nstrong),
        "smoothed_max" => Dict(string(h) => Dict("theta" => smax[h].theta, "value" => smax[h].value) for h in HS), "smoothed_max_location_shift_dex" => shift,
        "plateau80_pm0p1" => NM ? nothing : Dict(string(k) => v for (k, v) in pairs(p80)), "plateau50_pm0p1" => NM ? nothing : Dict(string(k) => v for (k, v) in pairs(p50)),
        "smoothed_at_theta_rep" => s_rep, "excess_of_N_max_over_smoothed" => NM ? nothing : A.M - s_rep,
        "excess_in_sqrt_units" => NM || s_rep == 0 ? nothing : (A.M - s_rep) / sqrt(s_rep),
        "smoothed_at_theta_max" => sm[PH][end],
        "bootstrap" => Dict("log10_theta_rep_p16_p50_p84" => fin.([bq.p16, bq.p50, bq.p84]), "N_max_p16_p84" => [qtl(bb.N_max, 0.16), qtl(bb.N_max, 0.84)],
                            "fraction_BOUNDARY_LIMITED" => mean(bb.status .== "BOUNDARY_LIMITED"), "fraction_below_1e-5" => mean(coalesce.(bb.theta_rep .< 1e-5, false))),
        "tight_lower_edge_sensitivity" => edge, "min_theta_90" => fin(minimum(t90)), "min_theta_100" => fin(minimum(t100)),
        "N_theta_90_le_0p5" => count(t90 .<= TH_MAX), "N_theta_100_le_0p5_overabundant_at_theta_max" => count(t100 .< TH_MAX),
        "spearman_h11_vs_log_theta_90" => fin(rho), "n_PRE_dominated" => count(pdom))
    # keys that exist only when the condition occurs (the reference-case outputs stay unchanged)
    nzero > 0 && (summ[lab]["n_geometries_Omega_zero"] = nzero)
    any(==("NO_MATCH"), bb.status) && (summ[lab]["bootstrap"]["fraction_NO_MATCH"] = mean(bb.status .== "NO_MATCH"))
    n < 20 && (summ[lab]["bootstrap"]["note"] = "fewer than 20 geometries: the bootstrap distribution has few distinct values and is not informative")
    mi(x) = x === nothing || (x isa AbstractFloat && isnan(x)) ? missing : x
    push!(sum_rows, (scenario=lab, T_reh_GeV=TREH[lab], N_tight_max=A.M, f_tight_max=A.M / n, n_maximal_intervals=length(A.ivs),
        theta_rep=mi(A.rep), status=st.status, robust_preferred_angle=robust, N_tight_at_rep=NM ? missing : c["TIGHT_DM"],
        N_underabundant_at_rep=NM ? missing : c["ALLOWED_UNDERABUNDANT"], N_overabundant_at_rep=NM ? missing : c["OVERABUNDANT"],
        N_strongly_underabundant_at_rep=mi(nstrong), f_allowed_at_rep=NM ? missing : (c["TIGHT_DM"] + c["ALLOWED_UNDERABUNDANT"]) / n,
        smoothed_max_theta=mi(smax[PH].theta), smoothed_max_value=smax[PH].value, smoothed_max_shift_dex=mi(shift),
        plateau80_from=NM ? missing : p80.from, plateau80_to=NM ? missing : p80.to, plateau80_width_dex=NM ? missing : p80.width_dex,
        excess_sqrt_units=NM || s_rep == 0 ? missing : (A.M - s_rep) / sqrt(s_rep),
        boot_log10_theta_p16=mi(bq.p16), boot_log10_theta_p50=mi(bq.p50), boot_log10_theta_p84=mi(bq.p84), boot_frac_boundary_limited=mean(bb.status .== "BOUNDARY_LIMITED"),
        N_max_tight_lower_0p8=edge["0.8"]["N_max"], theta_rep_tight_lower_0p8=mi(edge["0.8"]["theta_rep"]),
        N_max_tight_lower_0p95=edge["0.95"]["N_max"], theta_rep_tight_lower_0p95=mi(edge["0.95"]["theta_rep"]),
        min_theta_90=minimum(t90), min_theta_100=minimum(t100), N_theta_90_le_0p5=count(t90 .<= TH_MAX), descriptor=descriptor); promote=true)
    # ---- h11 dependence ----
    for h in sort(unique(G.h11))
        s = G.h11 .== h; l = log10.(t90[s])
        push!(h11_rows, (scenario=lab, h11=h, n_geometries=count(s), log10_theta_90_p16=qtl(l, 0.16), log10_theta_90_median=med(l), log10_theta_90_p84=qtl(l, 0.84),
            n_theta_90_le_0p5=count(t90[s] .<= TH_MAX), frac_theta_90_le_0p5=mean(t90[s] .<= TH_MAX), n_overabundant_at_theta_max=count(t100[s] .< TH_MAX),
            frac_PRE_dominated=mean(pdom[s]), n_tight_at_rep=NM ? missing : count(isequal("TIGHT_DM"), cls[s]),
            n_underabundant_at_rep=NM ? missing : count(isequal("ALLOWED_UNDERABUNDANT"), cls[s]), n_overabundant_at_rep=NM ? missing : count(isequal("OVERABUNDANT"), cls[s]),
            n_strongly_underabundant_at_rep=NM ? missing : count(A.rep .< t10[s]), median_log10_Omega_ref=med(log10.(om[s]))); promote=true)
    end
    for b in ("01-03", "04-10", "11-20", "21-30", "31-40", "41-50", "51+")
        s = hbin.(G.h11) .== b; any(s) || continue           # bins without geometries are not reported
        l = log10.(t90[s])
        push!(hb_rows, (scenario=lab, h11_bin=b, n_geometries=count(s), log10_theta_90_median=med(l), frac_theta_90_le_0p5=mean(t90[s] .<= TH_MAX),
            frac_tight_at_rep=NM ? missing : mean(isequal("TIGHT_DM"), cls[s]), frac_underabundant_at_rep=NM ? missing : mean(isequal("ALLOWED_UNDERABUNDANT"), cls[s]),
            frac_overabundant_at_rep=NM ? missing : mean(isequal("OVERABUNDANT"), cls[s]), frac_overabundant_at_theta_max=mean(t100[s] .< TH_MAX), frac_PRE_dominated=mean(pdom[s])); promote=true)
    end
end
csvw(joinpath(OUT, "theta_geometry_final.csv"), geo_rows)
csvw(joinpath(OUT, "theta_summary_final.csv"), sum_rows)
csvw(joinpath(OUT, "theta_per_h11_final.csv"), h11_rows)
csvw(joinpath(OUT, "theta_per_h11bin_final.csv"), hb_rows)
csvw(joinpath(OUT, "theta_exact_overlap_intervals.csv"), iv_rows)
csvw(joinpath(OUT, "theta_threshold_quantiles.csv"), q_rows)
csvw(joinpath(OUT, "theta_bootstrap.csv"), boot_rows)
summ["global"] = Dict("min_theta_90_all_scenarios" => fin(global_min90), "domain" => CFG["theta"]["domain"], "theta_min" => nothing,
                      "Omega_DM_h2" => ODM, "TIGHT_DM" => [X_LO, X_HI], "tight_window_width_dex" => W)
summ["population"] = POP
jsonw(joinpath(OUT, "theta_summary_final.json"), summ)
println("\npopulation ", POP)
nrow(sum_rows) > 0 && show(sum_rows[:, [:scenario, :N_tight_max, :theta_rep, :status, :robust_preferred_angle, :N_overabundant_at_rep, :plateau80_from, :plateau80_to, :smoothed_max_shift_dex, :boot_log10_theta_p16, :boot_log10_theta_p84, :min_theta_90]], allcols=true)
println("\nglobal minimum theta_90 = ", global_min90)
return summ, nsize
end

const PRIM = joinpath(RES, "theta"); const SENS = joinpath(RES, "theta_sensitivity_partial_validity")
SP, NP = analyse(PRIM, "PRIMARY", ==("PRIMARY"))
SS, NS = analyse(SENS, "SENSITIVITY_INCLUDING_PARTIAL_VALIDITY", p -> p in ("PRIMARY", "SENSITIVITY_ONLY"))
# every geometry with its class and population membership; denominators and headline results of both populations
prow = DataFrame()
pblock(s, ns, lab) = get(s[lab], "status", "") == "EMPTY_POPULATION" ? Dict{String,Any}("n_geometries" => 0, "status" => "EMPTY_POPULATION") :
    Dict{String,Any}("n_geometries" => ns[lab], "N_tight_max" => s[lab]["N_tight_max"], "f_tight_max" => s[lab]["f_tight_max"],
         "maximal_intervals" => s[lab]["maximal_intervals"], "theta_rep" => s[lab]["theta_rep"], "status" => s[lab]["upper_boundary"]["status"],
         "descriptor" => s[lab]["descriptor"])
psum = Dict{String,Any}("definition" => "primary = EVALUATED_POSITIVE + EVALUATED_ZERO (complete abundances); sensitivity = primary + PARTIAL_VALIDITY " *
    "(partial sums over valid modes, not complete abundances); INFLATION_INVALID_ALL: abundance not evaluated, in neither", "scenarios" => Dict{String,Any}())
for lab in LABS
    V = validity(lab)
    for r in eachrow(V)
        push!(prow, (scenario=lab, geometry_id=r.geometry_id, h11=r.h11, abundance_validity=r.abundance_validity, reason=r.reason,
            number_of_modes=r.number_of_modes, number_included=r.number_included, number_inflation_excluded=r.number_inflation_excluded,
            theta_population=r.theta_population, in_primary=r.theta_population == "PRIMARY", in_sensitivity=r.theta_population != "EXCLUDED_NOT_EVALUATED",
            Omega_complete_h2_at_theta_ref=r.Omega_complete_h2_at_theta_ref, Omega_valid_modes_partial_h2_at_theta_ref=r.Omega_valid_modes_partial_h2_at_theta_ref); promote=true)
    end
    cnt = Dict(c => count(==(c), V.abundance_validity) for c in ("EVALUATED_POSITIVE", "EVALUATED_ZERO", "PARTIAL_VALIDITY", "INFLATION_INVALID_ALL"))
    pp = pblock(SP, NP, lab); ss = pblock(SS, NS, lab)
    pp["n_geometries"] == cnt["EVALUATED_POSITIVE"] + cnt["EVALUATED_ZERO"] || error("$lab: primary denominator differs from the class counts")
    ss["n_geometries"] == pp["n_geometries"] + cnt["PARTIAL_VALIDITY"] || error("$lab: sensitivity denominator differs from the class counts")
    psum["scenarios"][lab] = Dict("total_geometries" => nrow(V), "by_abundance_validity" => cnt, "primary" => pp, "sensitivity_partial_validity" => ss,
        "excluded_not_evaluated" => Dict("n_geometries" => cnt["INFLATION_INVALID_ALL"],
                                         "geometry_ids" => V.geometry_id[V.abundance_validity .== "INFLATION_INVALID_ALL"]))
end
csvw(joinpath(PRIM, "theta_population.csv"), prow)
jsonw(joinpath(PRIM, "theta_population_summary.json"), psum)
