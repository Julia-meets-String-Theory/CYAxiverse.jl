# Shared code for the final small-angle (theta) analysis (Julia 1.8.4; environment environment/julia_analysis).
# Read-only with respect to every input; writes only inside the theta output directory.
# Locations (no machine-specific paths): WORK/relic (relic run: run_config.json, tables/) and WORK/final_analysis
# (outputs: theta/, verification/). Override with SHOROQ_RELIC_DIR and SHOROQ_FINAL_DIR.
using CSV, DataFrames, Statistics, Printf, SHA, JSON, Random, Dates

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const CFG = JSON.parsefile(get(ENV, "SHOROQ_THETA_CONFIG", joinpath(get(ENV, "SHOROQ_FINAL_DIR", joinpath(ENV["WORK"], "final_analysis")), "theta_config.json")))
const RELIC = get(ENV, "SHOROQ_RELIC_DIR", joinpath(ENV["WORK"], "relic"))
const PCFG = JSON.parsefile(joinpath(RELIC, "run_config.json"))
# a run configuration holds absolute paths of the WORK it was written in: a moved or copied WORK would silently read the other one
(p -> ispath(p) ? realpath(p) : normpath(p))(PCFG["run"]["production_output_dir"]) == realpath(RELIC) ||
    error("run_config.json was written for $(PCFG["run"]["production_output_dir"]), but the relic directory is $(realpath(RELIC)): " *
          "the WORK directory was moved or copied; rerun the relic stage in the new location (outputs are not relocatable)")
const FI = Dict{String,Any}(
    "production_dir" => RELIC,
    "production_config" => joinpath(RELIC, "run_config.json"),
    "production_script" => joinpath(REPO, "engine", "relic", "compute_relic_abundance.jl"),
    "canonical_mode_table" => PCFG["input"]["canonical_mode_table"],
    "canonical_mode_table_sha256" => PCFG["input"]["canonical_mode_table_sha256"],
    "relic_input_dir" => PCFG["input"]["input_dir"],
    "relic_input_manifest" => PCFG["input"]["manifest"],
    "relic_input_manifest_sha256" => PCFG["input"]["manifest_sha256"],
    "relic_input_geometries_csv" => joinpath(PCFG["input"]["input_dir"], "relic_abundance_input_geometries.csv"),
    "relic_input_SHA256SUMS" => PCFG["input"]["input_sha256sums_file"],
    "relic_input_SHA256SUMS_sha256" => PCFG["input"]["input_sha256sums_file_sha256"],
    "gstar_table_sha256" => PCFG["cosmology"]["gstar_table_sha256"])
const PROD = FI["production_dir"]; const PTAB = joinpath(PROD, "tables")
const RES = get(ENV, "SHOROQ_FINAL_DIR", joinpath(ENV["WORK"], "final_analysis"))
const FA = RES
const LABS = [String(s["label"]) for s in PCFG["reheating"]["scenarios"]]          # scenarios of the relic run, in order
const TREH = Dict(String(s["label"]) => Float64(s["T_reh_GeV"]) for s in PCFG["reheating"]["scenarios"])
const ODM = Float64(CFG["dark_matter"]["Omega_DM_h2"])
const X_LO = Float64(CFG["dark_matter"]["TIGHT_DM"][1]); const X_HI = Float64(CFG["dark_matter"]["TIGHT_DM"][2])
const TH_REF = Float64(CFG["theta"]["theta_ref"]); const TH_MAX = Float64(CFG["theta"]["theta_max"])
CFG["theta"]["theta_min"] === nothing || error("the final analysis has no lower theta cutoff")
const W = log10(sqrt(X_HI / X_LO))                     # common TIGHT window width in log10(theta)

sha(p) = bytes2hex(open(sha256, p))
csvw(path, df) = (mkpath(dirname(path)); CSV.write(path, df); println("wrote ", relpath(path, FA)))
jsonw(path, x) = (mkpath(dirname(path)); open(io -> JSON.print(io, x, 1), path, "w"); println("wrote ", relpath(path, FA)))
# column types from the output schema, not from inference: %.17g writes an integral float such as 1.19e16 without a
# decimal point, and a small table may hold only such values in a column (Int64 inference would then overflow f^2)
# methodology v2 adds the column w_R to every table (configs/relic_output_schema_v2.json); a v1 run configuration has no methodology_version
const METHOD_VERSION = Int(get(PCFG, "methodology_version", 1))
METHOD_VERSION in (1, 2) || error("run configuration: unsupported methodology_version $(METHOD_VERSION) (1 or 2)")
const SCHEMA_TABLES = JSON.parsefile(joinpath(REPO, "configs", METHOD_VERSION == 1 ? "relic_output_schema.json" : "relic_output_schema_v2.json"))["tables"]
# constant pre-reheating equation of state of every scenario: fixed 0 in v1, the scenario's w_R in v2
const WR = Dict(String(s["label"]) => (METHOD_VERSION == 1 ? 0.0 : Float64(s["w_R"])) for s in PCFG["reheating"]["scenarios"])
# v2 theta outputs carry w_R next to every scenario label (v1 outputs are unchanged)
tagw!(df) = (METHOD_VERSION >= 2 && nrow(df) > 0 && (df.w_R = [WR[l] for l in df.scenario]); df)
floatcols(t) = Dict(Symbol(c["name"]) => Float64 for c in SCHEMA_TABLES[t] if c["type"] == "float")
geometry(lab) = CSV.read(joinpath(PTAB, "geometry_summary_$(lab).csv"), DataFrame; missingstring="NA", types=floatcols("geometry_summary_<label>.csv"))
master(lab) = CSV.read(joinpath(PTAB, "master_modes_$(lab).csv"), DataFrame; missingstring="NA", types=floatcols("master_modes_<label>.csv"))
inputs_csv() = CSV.read(FI["relic_input_geometries_csv"], DataFrame; missingstring="")
validity(lab) = CSV.read(joinpath(PTAB, "geometry_abundance_validity_$(lab).csv"), DataFrame; missingstring="NA",
                         types=floatcols("geometry_abundance_validity_<label>.csv"))

"""Physics block of the production script, evaluated byte-for-byte in its own module (checksum-pinned)."""
function load_physics()
    p = FI["production_script"]; src = read(p, String)
    a = findfirst("# ============================== physics block", src); b = findfirst("# ============================== end of physics block", src)
    (a === nothing || b === nothing) && error("physics block markers not found")
    M = Module(:ProductionPhysics); Core.eval(M, :(using DelimitedFiles))
    Base.include_string(M, src[first(a):prevind(src, first(b))], "production_physics_block")
    tabpath = PCFG["cosmology"]["gstar_table"]; sha(tabpath) == FI["gstar_table_sha256"] || error("g* table checksum mismatch")
    tab = Base.invokelatest(M.thermal_table, tabpath)
    return M, tab
end

theta_at(om, x) = TH_REF .* sqrt.(x .* ODM ./ om)

# ---------------- exact closed-interval machinery (identical logic to the validated theta_core.py) ----------------
function exact_profile(a, b, lo, hi)
    a2 = max.(a, lo); b2 = min.(b, hi); keep = a2 .<= b2
    S = sort(a2[keep]); E = sort(b2[keep])
    P = sort(unique(vcat(S, E, [lo, hi])))
    vpt = [searchsortedlast(S, p) - (searchsortedfirst(E, p) - 1) for p in P]
    vseg = [searchsortedlast(S, P[i]) - searchsortedlast(E, P[i]) for i in 1:length(P)-1]
    return P, vpt, vseg
end
function maximal_set(P, vpt, vseg)
    M = max(maximum(vpt), isempty(vseg) ? 0 : maximum(vseg)); ivs = Vector{Vector{Float64}}(); cur = nothing
    for i in eachindex(P)
        atoms = [(vpt[i], P[i], P[i])]; i <= length(vseg) && push!(atoms, (vseg[i], P[i], P[i+1]))
        for (val, l, r) in atoms
            if val == M
                cur = cur === nothing ? [l, r] : [cur[1], r]
            elseif cur !== nothing
                push!(ivs, cur); cur = nothing
            end
        end
    end
    cur !== nothing && push!(ivs, cur)
    return M, ivs
end
function representative(ivs)
    w(iv) = iv[1] > 0 ? iv[2] / iv[1] : Inf
    iv = ivs[argmax(map(w, ivs))]
    return iv[1] > 0 ? sqrt(iv[1] * iv[2]) : iv[2]
end
"""N(theta) of the exact step function at arbitrary theta."""
function evaluate_step(P, vpt, vseg, th)
    map(th) do t
        i = clamp(searchsortedlast(P, t), 1, length(P))
        P[i] == t ? vpt[i] : vseg[clamp(i, 1, length(vseg))]
    end
end
"""Exact class labels at one theta from the analytic thresholds."""
classify_at(th, t90, t100) = [th > b ? "OVERABUNDANT" : th >= a ? "TIGHT_DM" : "ALLOWED_UNDERABUNDANT" for (a, b) in zip(t90, t100)]
"""Exact smoothed count: mean of N over |log10 theta' - log10 theta| <= h = (1/2h) sum_g overlap([l_g, l_g + W], window)."""
function smoothed(l90sorted, L; h=0.1, w=W)
    out = similar(collect(Float64, L))
    for (k, l) in enumerate(L)
        i1 = searchsortedfirst(l90sorted, l - h - w); i2 = searchsortedlast(l90sorted, l + h)
        s = 0.0
        @inbounds for j in i1:i2
            x = l90sorted[j]; s += max(0.0, min(x + w, l + h) - max(x, l - h))
        end
        out[k] = s / (2h)
    end
    return out
end
"""Quantile (type 7). Values may be +Inf (theta thresholds of a geometry with Omega = 0, which never reaches a band) or -Inf
(log10 of Omega = 0): interpolation towards an infinite neighbour gives that infinity. NaN only for an empty vector."""
function qtl(v, p)
    isempty(v) && return NaN
    all(isfinite, v) && return quantile(v, p)                # unchanged path for finite data
    s = sort(collect(Float64, v)); h = (length(s) - 1) * p + 1; lo = floor(Int, h); hi = min(lo + 1, length(s)); g = h - lo
    a, b = s[lo], s[hi]
    a == b ? a : g == 0 ? a : isinf(b) ? b : isinf(a) ? a : a + g * (b - a)
end
med(v) = all(isfinite, v) ? median(v) : qtl(v, 0.5)   # Statistics.median (unchanged) for finite data
fin(x) = x === nothing || x === missing || !isfinite(x) ? nothing : x      # JSON.jl writes Inf and NaN as null: say so explicitly
