# STEP 0: verify that the relic run and its inputs are complete and consistent before the theta analysis. Nothing is repaired:
# any failure stops the run. General checks run for every sample; population counts (and the Shoroq spectrum layout)
# are checked only when the theta configuration carries expectations (block "expected", a reference case).
include(joinpath(@__DIR__, "common.jl"))
using HDF5
checks = Vector{Dict{String,Any}}()
function check!(name, ok, detail="")
    push!(checks, Dict("check" => name, "pass" => Bool(ok), "detail" => string(detail)))
    @printf("%-4s %s %s\n", ok ? "PASS" : "FAIL", name, detail)
end
function sumsfile_ok(path; base=dirname(path))
    bad = String[]; n = 0
    for ln in eachline(path)
        isempty(strip(ln)) && continue
        h, f = split(ln, limit=2); f = strip(f); f = startswith(f, "*") ? f[2:end] : f
        p = isabspath(f) ? f : normpath(joinpath(base, f)); n += 1
        (isfile(p) && sha(p) == h) || push!(bad, f)
    end
    return n, bad
end

# ---- 1. checksums and provenance of the relic run (run_config.json written by engine/relic/02_write_run_config.py) ----
check!("canonical mode table sha256 = run_config pin", sha(FI["canonical_mode_table"]) == FI["canonical_mode_table_sha256"])
check!("relic input manifest sha256 = run_config pin", sha(FI["relic_input_manifest"]) == FI["relic_input_manifest_sha256"])
check!("relic input SHA256SUMS file sha256 = run_config pin", sha(FI["relic_input_SHA256SUMS"]) == FI["relic_input_SHA256SUMS_sha256"])
n, bad = sumsfile_ok(FI["relic_input_SHA256SUMS"]); check!("relic input SHA256SUMS ($(n) files)", isempty(bad), join(bad, ";"))
meta = JSON.parsefile(joinpath(PTAB, "run_metadata.json"))
check!("run_metadata: manifest/config/script/mode-table checksums", meta["manifest_sha256"] == FI["relic_input_manifest_sha256"] &&
       meta["config_sha256"] == sha(FI["production_config"]) && meta["script_sha256"] == sha(FI["production_script"]) &&
       meta["mode_table_sha256"] == FI["canonical_mode_table_sha256"] && meta["gstar_table_sha256"] == FI["gstar_table_sha256"])
const EXPECT = haskey(CFG, "expected")
EX = EXPECT ? CFG["expected"] : Dict{String,Any}()
NROWS = length(readlines(FI["canonical_mode_table"])) - 1
check!("relic run covers every row of the canonical mode table", meta["rows"] == NROWS, "$(meta["rows"]) / $(NROWS)")
EXPECT && check!("production run approved and full population", meta["production_run_approved"] == true && meta["is_full_population"] == true && meta["rows"] == EX["modes"])

# ---- 2. population and spectrum source (every spectrum file re-hashed and re-read) ----
G = inputs_csv(); G = G[in.(G.status, Ref(["ORIGINAL_SPECTRUM", "VACUUM_CORRECTED"])), :]     # usable geometries (EXCLUDED_* carry no spectrum)
NG = nrow(G); NM = sum(G.n_modes_above_H0)
check!("canonical mode table rows = modes of the usable manifest geometries", NROWS == NM, "$(NROWS) / $(NM)")
if EXPECT
    check!("manifest geometries = $(EX["geometries"])", NG == EX["geometries"], NG)
    check!("manifest modes = $(EX["modes"])", NM == EX["modes"], NM)
    vs = combine(groupby(G, :status), nrow => :n, :n_modes_above_H0 => sum => :modes)
    for (st, ng) in EX["geometries_by_vacuum_status"]
        r = vs[vs.status .== st, :]
        check!("manifest $(st): geometries/modes", nrow(r) == 1 && r.n[1] == ng && r.modes[1] == EX["modes_by_vacuum_status"][st], "$(r.n) / $(r.modes)")
    end
end
# spectrum_source is absolute (frozen 2026-09-30 manifest) or relative to the manifest's spectrum_source_base (clean pipeline)
MAN = JSON.parsefile(FI["relic_input_manifest"])
SRCB = normpath(joinpath(FI["relic_input_dir"], get(MAN, "spectrum_source_base", "")))
src(s) = isabspath(s) ? s : joinpath(SRCB, s)
G.spectrum_source = src.(G.spectrum_source)
if EXPECT                                   # layout of the producer pipeline (reference case)
    corr = G[G.status .== "VACUUM_CORRECTED", :]
    check!("$(nrow(corr)) VACUUM_CORRECTED read from the vacuum-repair corrected spectra", all(occursin("corrected_spectra/", s) for s in corr.spectrum_source))
    orig = G[G.status .== "ORIGINAL_SPECTRUM", :]
    check!("$(nrow(orig)) ORIGINAL read from the theta = 0 production spectra", all(!occursin("corrected_spectra/", s) && occursin("spectra/", s) for s in orig.spectrum_source))
end
view_ok = all(realpath(joinpath(FI["relic_input_dir"], "spectra", id, "cyax.h5")) == realpath(s) for (id, s) in zip(G.geometry_id, G.spectrum_source))
check!("relic input spectra view resolves to the manifest spectrum_source", view_ok)
badsha = [id for (id, s, h) in zip(G.geometry_id, G.spectrum_source, G.spectrum_source_sha256) if sha(s) != h]
check!("all $(NG) spectrum files match their manifest sha256", isempty(badsha), join(badsha[1:min(end, 5)], ";"))

MS = Dict(lab => master(lab) for lab in LABS); M5 = MS[LABS[1]]
check!("master tables: $(NM) rows each", all(nrow(m) == NM for m in values(MS)))
modecols = [:geometry_id, :h11, :mode_index, :mass_eV, :log10_mass_eV, :f_pert_GeV, :log10_f_pert_GeV, :vacuum_status, :inflation_valid, :above_mKK, :above_fpert]
check!("mode-level columns identical across the $(length(LABS)) scenarios", all(isequal(M5[:, modecols], m[:, modecols]) for m in values(MS)))
key = Dict((r.geometry_id, r.mode_index) => i for (i, r) in enumerate(eachrow(M5)))
nmis = 0; maxdm = 0.0; maxdf = 0.0; nvol = 0; seen = 0
GS = geometry(LABS[1]); vol = Dict(zip(GS.geometry_id, GS.cy_volume))
for r in eachrow(G)
    h5open(r.spectrum_source, "r") do f
        m = read(f["spectrum/physical/m"]); fp = read(f["spectrum/physical/fpert_log10"]); mi = read(f["spectrum/physical/mode_indices"])
        V = read(f["cytools/geometric/CY_volume"])
        global seen += length(m)
        length(m) == r.n_modes_above_H0 || (global nmis += 1)
        for (j, idx) in enumerate(mi)
            i = get(key, (r.geometry_id, Int(idx)), 0)
            if i == 0
                global nmis += 1
            else
                global maxdm = max(maxdm, abs(M5.log10_mass_eV[i] - m[j])); global maxdf = max(maxdf, abs(M5.log10_f_pert_GeV[i] - fp[j]))
                M5.vacuum_status[i] == r.status || (global nmis += 1)
            end
        end
        abs(vol[r.geometry_id] / V - 1) < 1e-12 || (global nvol += 1)
    end
end
check!("every production mode is present in its HDF5 spectrum ($(NM) read)", nmis == 0 && seen == NM, "missing/mismatched=$(nmis), read=$(seen)")
check!("production log10 m and log10 f_pert equal the HDF5 spectrum values", maxdm <= 1e-12 && maxdf <= 1e-12, @sprintf("max |dlog m|=%.2e, |dlog f|=%.2e", maxdm, maxdf))
check!("production CY volume equals cytools/geometric/CY_volume", nvol == 0, nvol)
EXPECT && check!("vacuum_status per mode: $(EX["modes_by_vacuum_status"]["ORIGINAL_SPECTRUM"]) ORIGINAL / $(EX["modes_by_vacuum_status"]["VACUUM_CORRECTED"]) CORRECTED",
    count(==("ORIGINAL_SPECTRUM"), M5.vacuum_status) == EX["modes_by_vacuum_status"]["ORIGINAL_SPECTRUM"] && count(==("VACUUM_CORRECTED"), M5.vacuum_status) == EX["modes_by_vacuum_status"]["VACUUM_CORRECTED"])

# ---- 3. inflation-validity exclusion (per mode, Fuzzy fn 16) ----
three_HI = Float64(PCFG["validity"]["inflation_cut"]["three_H_I_max_GeV"])
IC = PCFG["validity"]["inflation_cut"]; NX = get(IC, "expected_modes", nothing)
for lab in LABS
    M = MS[lab]
    ex = M[.!M.inflation_valid, :]
    ok = all(==("PRE_REHEATING"), ex.category) &&
         all(==("INFLATIONARY_VALIDITY_EXCLUDED"), ex.abundance_state) && all(ismissing, ex.Omega_h2_at_theta_ref) && !any(ex.included_in_cdm_sum) &&
         all(1e-9 .* ex.mass_eV .> three_HI) && all(1e-9 .* M[M.inflation_valid, :mass_eV] .<= three_HI)
    check!("$(lab): inflation-excluded modes are PRE, NA abundance, not summed; cut exactly at 3H_I,max", ok, nrow(ex))
    NX === nothing || check!("$(lab): $(NX) inflation-excluded modes in $(IC["expected_geometries"]) geometries", nrow(ex) == NX && length(unique(ex.geometry_id)) == IC["expected_geometries"], nrow(ex))
    inc_rule = M.included_in_cdm_sum .== ((M.category .∈ Ref(["POST_REHEATING", "PRE_REHEATING"])) .& M.inflation_valid)
    check!("$(lab): inclusion = (POST or PRE) and inflation-valid, nothing else", all(inc_rule))
    g = geometry(lab); sums = combine(groupby(M[M.included_in_cdm_sum, :], :geometry_id), :Omega_h2_at_theta_ref => sum => :s)
    d = Dict(zip(sums.geometry_id, sums.s))
    check!("$(lab): Omega_geometry = sum of included modes; excluded count = master table", sum(g.number_inflation_excluded) == nrow(ex) &&
        all(ismissing(g.Omega_geometry_h2_at_theta_ref[i]) || g.Omega_geometry_h2_at_theta_ref[i] == 0 ? !haskey(d, g.geometry_id[i]) :
            abs(g.Omega_geometry_h2_at_theta_ref[i] / d[g.geometry_id[i]] - 1) < 1e-12 for i in 1:nrow(g)) && nrow(g) == NG)
    # every usable geometry of the relic input is in the geometry table (also without mode above H0), Omega evaluated (0 or > 0)
    om_ = g.Omega_geometry_h2_at_theta_ref; nas = ismissing.(om_); z = .!nas .& (coalesce.(om_, -1.0) .== 0)
    check!("$(lab): geometry table = usable relic-input geometries; Omega 0 iff evaluated empty sum, NA iff nothing included and a mode not evaluated",
        Set(g.geometry_id) == Set(G.geometry_id) && all(nas .== ((g.number_included .== 0) .& (g.number_inflation_excluded .> 0))) &&
        all(z .== ((g.number_included .== 0) .& (g.number_inflation_excluded .== 0))), "Omega 0: $(count(z)), NA: $(count(nas))")
    V = validity(lab)
    check!("$(lab): validity table rows = geometry rows; classes consistent with the counts",
        V.geometry_id == g.geometry_id && all(r -> r.abundance_validity == (r.number_inflation_excluded == 0 ? (r.number_included > 0 ? "EVALUATED_POSITIVE" : "EVALUATED_ZERO") :
            r.number_inflation_excluded == r.number_of_modes ? "INFLATION_INVALID_ALL" : "PARTIAL_VALIDITY"), eachrow(V)))
    # ---- 4. no 100 eV window ----
    n100 = count(M.included_in_cdm_sum .& (M.mass_eV .> 100))
    check!("$(lab): no 100 eV cut (methodology flag off; included modes above 100 eV: $(n100))", PCFG["methodology_flags"]["light_window_100eV"] == false && (!EXPECT || n100 > 0))
    if haskey(PCFG, "expected_census_before_inflation_cut")
        exp = PCFG["expected_census_before_inflation_cut"][lab]
        check!("$(lab): census FROZEN/LATE/POST/PRE equals the approved values", all(count(==(c), M.category) == exp[c] for c in ("FROZEN", "LATE", "POST_REHEATING", "PRE_REHEATING")))
    end
end

# ---- 5. f_relic = f_pert with C = 1, physics and cosmology unchanged (abundances recomputed from the production physics block) ----
P, tab = load_physics()
check!("config: f_convention f_pert, f_relic/f_pert = 1", PCFG["axion"]["f_convention"] == "f_pert" && PCFG["axion"]["f_relic_over_f_pert"] == 1.0)
relq(a, b) = abs(a / b - 1)          # float literals such as 2.13*0.7e-33 differ from 1.491e-33 in the last bit (2.2e-16), as recorded by production
check!("H0 = 1.491e-33 eV (physics block, config, metadata; rel <= 1e-15)", relq(P.H0_GEV * 1e9, PCFG["cosmology"]["H0_eV"]) <= 1e-15 && PCFG["cosmology"]["H0_eV"] == get(EX, "H0_eV", 1.491e-33) &&
       meta["constants_checked_against_config"]["H0_eV"]["config"] == 1.491e-33)
check!("omega_m (rel <= 1e-15), theta_ref and w_R unchanged", relq(P.OMEGA_M_H2, PCFG["cosmology"]["omega_m"]) <= 1e-15 && P.THETA_INITIAL == TH_REF && P.W_REHEAT == 0.0)
maxrel = Dict{String,Float64}()
for lab in LABS
    M = MS[lab]
    sc = only(s for s in PCFG["reheating"]["scenarios"] if s["label"] == lab); treh = Float64(sc["T_reh_GeV"])
    rel = 0.0; relk = 0.0
    for r in eachrow(M)
        r.included_in_cdm_sum || continue
        if r.category == "PRE_REHEATING"
            w = WR[lab]
            if w == 0.0      # v1 and v2 at w_R = 0: the frozen physics block itself
                om = Base.invokelatest(P.pre_reheating_abundance, tab, r.log10_mass_eV, r.log10_f_pert_GeV, treh).omega
                relk = max(relk, abs(r.Omega_h2_at_theta_ref / (r.f_pert_GeV^2 * TH_REF^2) / sc[METHOD_VERSION == 1 ? "Omega_pre_h2_per_fGeV2_theta2" : "Omega_pre_h2_per_fGeV2_theta2_at_m_eq_3H_R"] - 1))
            else             # v2, w_R != 0: direct power form (2.15)+(2.10) from the physics block's constants, independent of the engine's logarithmic evaluation
                mg = 10.0^(r.log10_mass_eV - 9); HR = Base.invokelatest(P.hubble_reheat, tab, treh); opz = Base.invokelatest(P.onepz_reheat, tab, treh) * (mg / (3HR))^(2 / (3 * (1 + w)))
                om = 0.5 * mg^2 * (10.0^r.log10_f_pert_GeV)^2 * TH_REF^2 / opz^3 / P.RHO_CRIT_OVER_H2_GEV4
                kexp = sc["Omega_pre_h2_per_fGeV2_theta2_at_m_eq_3H_R"] * (mg / (3 * sc["H_R_GeV"]))^sc["PRE_mass_exponent_2w_over_1pw"]   # relative form k(T_R) (m/3H_R)^(2w/(1+w))
                relk = max(relk, abs(r.Omega_h2_at_theta_ref / (r.f_pert_GeV^2 * TH_REF^2) / kexp - 1))
            end
        else
            om = Base.invokelatest(P.raw_abundance, tab, r.log10_mass_eV, r.log10_f_pert_GeV, r.T_osc_GeV).omega
        end
        rel = max(rel, abs(r.Omega_h2_at_theta_ref / om - 1))
    end
    maxrel[lab] = rel
    check!("$(lab): every included Omega reproduced from f_pert (C=1) by the production physics block", rel < 1e-12, @sprintf("max rel %.1e", rel))
    check!("$(lab): PRE Omega/(f_pert^2 theta_ref^2) equals the configured k(T_R)", relk < 1e-12, @sprintf("max rel %.1e", relk))
end
npass = count(c -> c["pass"], checks)
res = Dict("n_checks" => length(checks), "n_pass" => npass, "all_pass" => npass == length(checks), "checks" => checks,
           "input_checksums" => Dict(k => v for (k, v) in FI if endswith(k, "sha256")))
jsonw(joinpath(RES, "verification", "input_verification.json"), res)
npass == length(checks) || error("STOP: $(length(checks) - npass) input check(s) failed; nothing was repaired")
println("ALL $(npass) INPUT CHECKS PASS")
