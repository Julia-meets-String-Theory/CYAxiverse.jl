#!/usr/bin/env julia
# Relic abundance of every axion mode for the reheating scenarios of the run configuration (Julia 1.8.4, standard library only).
# This is the frozen 2026-10-01 production script (sha256 4483390956fa01709264842b589802d64b4355a13d3212cf0fa713628912143a)
# with these header comment lines rewritten to remove machine-specific paths and ONE driver line changed (G1, below: the
# population gate runs only when the configuration carries expected counts); the physics block is byte-identical
# (tests/test_static.py checks its checksum and that reverting G1 gives the frozen body). Physics block: legacy
# compute_tosc_abundance_hfull.jl (sha256 6af20481...) + approved edits E1-E6 + Fuzzy pre-reheating F1-F5; driver P1-P7.

using DelimitedFiles, Printf, Dates, SHA

# ============================== physics block (verbatim from the audited derived script) ==============================
const HBAR_GEV_S=6.582119569e-25
const MPL_REDUCED_GEV=2.435e18
const H100_GEV=(100.0/3.0856775814913673e19)*HBAR_GEV_S
const H0_GEV=2.13*0.7*1e-33*1e-9   # [E1] CYAxiverse generate.constants()["Hubble"] = 1.491e-33 eV, the VMM spectrum threshold
const HUBBLE_h=H0_GEV/H100_GEV      # [E1] derived, 0.698976
const T_CMB_GEV=2.3486541805581e-13
const OMEGA_M_H2=0.315*0.674^2      # [E2] physical matter density of the legacy cosmology, held fixed
const OMEGA_M0=OMEGA_M_H2/HUBBLE_h^2
# [E3] Omega_r0 and Omega_L0 are derived after the thermal table is loaded (see omega_r0 below)
const RHO_CRIT_OVER_H2_GEV4=3*MPL_REDUCED_GEV^2*H100_GEV^2
const THETA_INITIAL=pi/2
const T_SEARCH_MAX_GEV=1e18
const W_REHEAT=0.0   # [F1] Fuzzy (2.2),(2.15): reheating-era equation of state; scenario (2) modulus domination, w_R = 0

mutable struct ThermalTable
    T::Vector{Float64}; grho::Vector{Float64}; gs::Vector{Float64}
end
function thermal_table(path)
    raw=readdlm(path;comments=true,comment_char='#'); p=sortperm(Float64.(raw[:,1]))
    ThermalTable(Float64.(raw[p,1]),Float64.(raw[p,2]),Float64.(raw[p,4]))
end
function loginterp(T,x,y)
    T<=x[1] && return y[1]; T>=x[end] && return y[end]
    k=searchsortedlast(x,T); w=log(T/x[k])/log(x[k+1]/x[k])
    exp((1-w)*log(y[k])+w*log(y[k+1]))
end
grho(tab,T)=loginterp(T,tab.T,tab.grho)
gs(tab,T)=loginterp(T,tab.T,tab.gs)

# [E3] rho_r0 = (pi^2/30) g_rho(T0) T0^4 is physical; Omega_r0 = rho_r0/(3 M_pl^2 H0^2); flat closure for Omega_L0
omega_r0(tab)=(pi^2/30)*grho(tab,T_CMB_GEV)*T_CMB_GEV^4/(3*MPL_REDUCED_GEV^2*H0_GEV^2)
function expansion_terms(tab,T)
    gr0,gs0=grho(tab,T_CMB_GEV),gs(tab,T_CMB_GEV)
    radiation=omega_r0(tab)*(grho(tab,T)/gr0)*(T/T_CMB_GEV)^4
    matter=OMEGA_M0*(gs(tab,T)/gs0)*(T/T_CMB_GEV)^3
    (;radiation,matter,vacuum=1.0-OMEGA_M0-omega_r0(tab))
end
function Hfull(tab,T)
    x=expansion_terms(tab,T)
    H0_GEV*sqrt(x.radiation+x.matter+x.vacuum)
end
scale_factor(tab,T)=(T_CMB_GEV/T)*(gs(tab,T_CMB_GEV)/gs(tab,T))^(1/3)

function bisect_logT(f;lo=T_CMB_GEV,hi=T_SEARCH_MAX_GEV)
    flo,fhi=f(lo),f(hi)
    flo==0 && return lo; fhi==0 && return hi
    signbit(flo)==signbit(fhi) && return NaN
    for _ in 1:300
        mid=sqrt(lo*hi); fm=f(mid)
        if signbit(fm)==signbit(flo); lo=mid; flo=fm else hi=mid end
        abs(log(hi/lo))<1e-11 && break
    end
    sqrt(lo*hi)
end
function equality_temperature(tab)
    bisect_logT(T->begin
        x=expansion_terms(tab,T); log(x.radiation/x.matter)
    end)
end
function solve_tosc(tab,mass_gev)
    target=mass_gev/3
    target<=Hfull(tab,T_CMB_GEV) && return NaN
    Hfull(tab,T_SEARCH_MAX_GEV)<target && return Inf   # [E4] oscillates above the search ceiling: not frozen
    bisect_logT(T->log(Hfull(tab,T)/target))
end
function raw_abundance(tab,logm_ev,logf_gev,tosc)
    m=10.0^(logm_ev-9); f=10.0^logf_gev
    gr,sv=grho(tab,tosc),gs(tab,tosc)
    dilution=(gs(tab,T_CMB_GEV)/sv)*(T_CMB_GEV/tosc)^3
    omega=(0.5*m^2*f^2*THETA_INITIAL^2*dilution)/RHO_CRIT_OVER_H2_GEV4
    (;gr,sv,omega,residual=3Hfull(tab,tosc)/m-1)
end

# [F2] Fuzzy pre-reheating branch (arXiv:2412.12012 Sec. 2)
hubble_reheat(tab,treh)=sqrt((pi^2/30)*grho(tab,treh)*treh^4/(3*MPL_REDUCED_GEV^2))   # (2.2): rho_r(T_R)=3 H_R^2 M_pl^2
onepz_reheat(tab,treh)=1/scale_factor(tab,treh)                                        # (2.1) at z_R
function pre_reheating_abundance(tab,logm_ev,logf_gev,treh)
    m=10.0^(logm_ev-9); f=10.0^logf_gev; HR=hubble_reheat(tab,treh)
    onepz_a=onepz_reheat(tab,treh)*(m/(3HR))^(2/(3*(1+W_REHEAT)))                     # (2.15)
    omega=(0.5*m^2*f^2*THETA_INITIAL^2/onepz_a^3)/RHO_CRIT_OVER_H2_GEV4                 # (2.10)
    (;omega,a_osc=1/onepz_a,H_osc=m/3,HR)
end

# ============================== end of physics block ==============================

# =====================================================================================================================
# PRODUCTION DRIVER (relic_abundance_production_prep). Everything above the "production driver" banner is the physics
# block of the audited derived script relic_abundance_method_audit/scripts/derived/compute_tosc_abundance_fuzzy.jl,
# byte-for-byte (legacy compute_tosc_abundance_hfull.jl + E1-E6 + F1-F5). This driver only adds:
#   P1  input = A1 canonical mode table (manifest IDs), not per-h11 parameters_dataset.jls
#   P2  run configuration (JSON) checked against the physics constants and derived values before anything runs
#   P3  classification by thresholds 3H0, 3H(T_eq), 3H_R (identical to the audited branch conditions); the abundance
#       run re-derives LATE/POST from the solved T_osc and stops on any disagreement
#   P4  per-mode inflationary validity cut m_a > 3 H_I,max (Fuzzy footnote 16): INFLATIONARY_VALIDITY_EXCLUDED, not 0
#   P5  diagnostic flags above_mKK / above_fpert with ratios (never exclusions); recomputed and checked against A1
#   P6  output schema: per-mode master, per-geometry, per-h11, global summary, run metadata (NA never written as 0)
#   P7  census mode (classification + flags only, no abundance) and an approval guard for the full abundance run
# Usage: julia compute_relic_abundance_production.jl CONFIG.json MODE_TABLE.csv OUTPUT_DIR census|abundance
# =====================================================================================================================

# ---------------------------------------------------------------- minimal JSON (Julia 1.8 has no JSON stdlib)
mutable struct JReader; b::Vector{UInt8}; i::Int; end
jpeek(p)=p.b[p.i]
function jskip(p) while p.i<=length(p.b) && p.b[p.i] in (0x20,0x09,0x0a,0x0d); p.i+=1; end end
function jvalue(p)
    jskip(p); c=Char(jpeek(p))
    c=='{' && return jobject(p)
    c=='[' && return jarray(p)
    c=='"' && return jstring(p)
    if c=='t'; String(p.b[p.i:p.i+3])=="true" || error("json"); p.i+=4; return true end
    if c=='f'; String(p.b[p.i:p.i+4])=="false" || error("json"); p.i+=5; return false end
    if c=='n'; String(p.b[p.i:p.i+3])=="null" || error("json"); p.i+=4; return nothing end
    j=p.i
    while p.i<=length(p.b) && Char(p.b[p.i]) in "+-0123456789.eE"; p.i+=1; end
    t=String(p.b[j:p.i-1]); occursin(r"[.eE]",t) ? parse(Float64,t) : parse(Int,t)
end
function jobject(p)
    p.i+=1; d=Dict{String,Any}(); jskip(p)
    if Char(jpeek(p))=='}'; p.i+=1; return d end
    while true
        jskip(p); k=jstring(p); jskip(p); Char(jpeek(p))==':' || error("json: expected :"); p.i+=1
        d[k]=jvalue(p); jskip(p); c=Char(jpeek(p)); p.i+=1
        c=='}' && return d
        c==',' || error("json: expected , or }")
    end
end
function jarray(p)
    p.i+=1; v=Any[]; jskip(p)
    if Char(jpeek(p))==']'; p.i+=1; return v end
    while true
        push!(v,jvalue(p)); jskip(p); c=Char(jpeek(p)); p.i+=1
        c==']' && return v
        c==',' || error("json: expected , or ]")
    end
end
function jstring(p)
    Char(jpeek(p))=='"' || error("json: expected string"); p.i+=1; io=IOBuffer()
    while true
        c=p.b[p.i]; p.i+=1
        c==UInt8('"') && return String(take!(io))
        if c==UInt8('\\')
            e=Char(p.b[p.i]); p.i+=1
            if e=='u'; write(io,Char(parse(UInt16,String(p.b[p.i:p.i+3]);base=16))); p.i+=4
            else write(io, e=='n' ? '\n' : e=='t' ? '\t' : e=='r' ? '\r' : e=='b' ? '\b' : e=='f' ? '\f' : e) end
        else write(io,c) end
    end
end
json_read(path)=jvalue(JReader(read(path),1))
const OD=Vector{Pair{String,Any}}                       # ordered JSON object
jesc(s)=replace(replace(replace(String(s),"\\"=>"\\\\"),"\""=>"\\\""),"\n"=>"\\n")
function jwrite(io,x,ind=0)
    pad="\n"*" "^(ind+1)
    if x===nothing; print(io,"null")
    elseif x isa Bool; print(io,x ? "true" : "false")
    elseif x isa Integer; print(io,x)
    elseif x isa AbstractFloat; isfinite(x) || error("non-finite value in JSON output"); print(io,repr(Float64(x)))
    elseif x isa AbstractString; print(io,'"',jesc(x),'"')
    elseif x isa OD
        isempty(x) && return print(io,"{}")
        print(io,"{"); for (k,(a,b)) in enumerate(x); k>1 && print(io,","); print(io,pad,'"',jesc(a),"\": "); jwrite(io,b,ind+1); end; print(io,"\n"," "^ind,"}")
    elseif x isa AbstractDict
        jwrite(io,OD([string(k)=>x[k] for k in sort(collect(keys(x)),by=string)]),ind)
    elseif x isa Union{AbstractVector,Tuple}
        print(io,"["); for (k,v) in enumerate(x); k>1 && print(io,", "); jwrite(io,v,ind+1); end; print(io,"]")
    else error("cannot serialise $(typeof(x))") end
end
json_write(path,x)=open(io->(jwrite(io,x); println(io)),path,"w")

# ---------------------------------------------------------------- formatting / CSV
stamp()=Dates.format(now(),"yyyy-mm-dd HH:MM:SS")
fmtv(x)=x===nothing ? "NA" : x isa Bool ? (x ? "true" : "false") : x isa AbstractFloat ?
    (isfinite(x) ? @sprintf("%.17g",x) : error("non-finite value would be written")) : string(x)
function write_table(path,cols,rows)
    tmp=path*".building"
    open(tmp,"w") do io
        println(io,join(cols,','))
        for r in rows; println(io,join((fmtv(r[c]) for c in cols),',')); end
    end
    mv(tmp,path;force=true)
end
parsebool(s)=s in ("True","true") ? true : s in ("False","false") ? false : error("not a bool: $s")
filesha(path)=bytes2hex(open(sha256,path))
relerr(a,b)=a==b ? 0.0 : abs(a/b-1)

# ---------------------------------------------------------------- arguments, configuration, constant checks
length(ARGS)==4 || error("Usage: compute_relic_abundance_production.jl CONFIG.json MODE_TABLE.csv OUTPUT_DIR census|abundance")
const CONFIG_PATH=abspath(ARGS[1]); const TABLE_PATH=abspath(ARGS[2]); const OUTDIR=abspath(ARGS[3]); const MODE=ARGS[4]
MODE in ("census","abundance") || error("MODE must be census or abundance")
const CFG=json_read(CONFIG_PATH)
const COS=CFG["cosmology"]; const VAL=CFG["validity"]; const RUN=CFG["run"]
const THREE_HI_MAX_GEV=Float64(VAL["inflation_cut"]["three_H_I_max_GeV"])
const MPL_KK_GEV=Float64(VAL["mKK_flag"]["M_pl_GeV"])
const SCEN=[(String(s["label"]),Float64(s["T_reh_GeV"]),s) for s in CFG["reheating"]["scenarios"]]
# [G7] methodology version and per-scenario constant pre-reheating equation of state w_R (methodology v2). A v1 configuration has
# no methodology_version and no per-scenario w_R: every scenario then uses w_R = W_REHEAT = 0 through the frozen v1 function.
const METHOD_VERSION=Int(get(CFG,"methodology_version",1))
METHOD_VERSION in (1,2) || error("unsupported methodology_version $(get(CFG,"methodology_version",1)) (this script implements 1 and 2)")
function scenario_wR(lab,s)
    if METHOD_VERSION==1
        haskey(s,"w_R") && error("scenario $lab: w_R is fixed at 0 in methodology v1; use methodology v2 to choose it")
        return W_REHEAT
    end
    w=get(s,"w_R",CFG["reheating"]["w_R_default"])
    (w isa Real && !(w isa Bool) && isfinite(w) && -1<w<=1) ||
        error("scenario $lab: w_R must be a finite number with -1 < w_R <= 1 (H ~ a^(-3(1+w_R)/2) must fall with expansion; causal fluid), got $(repr(w))")
    Float64(w)
end
const WR_OF=Dict(lab=>scenario_wR(lab,s) for (lab,treh,s) in SCEN)
length(unique((treh,WR_OF[lab]) for (lab,treh,s) in SCEN))==length(SCEN) ||
    error("two scenarios share the same (T_reh_GeV, w_R): identical physics under two labels is ambiguous")
# [G7] constant-w_R PRE_REHEATING abundance, (2.10)+(2.15) evaluated in logarithms (no overflow of (1+z_a)^3). w_R == 0 calls the
# frozen v1 function, so a v2 run at w_R = 0 is bit-identical to v1. An abundance outside the float64 range stops the run: never clamped.
function pre_reheating_abundance_w(tab,logm_ev,logf_gev,treh,w)
    w==0.0 && return pre_reheating_abundance(tab,logm_ev,logf_gev,treh)
    m=10.0^(logm_ev-9); HR=hubble_reheat(tab,treh)
    lnopz_a=log(onepz_reheat(tab,treh))+(2/(3*(1+w)))*log(m/(3HR))                                   # (2.15)
    lnom=log(0.5)+2*log(m)+2*(logf_gev*log(10.0))+2*log(THETA_INITIAL)-log(RHO_CRIT_OVER_H2_GEV4)-3*lnopz_a   # (2.10)
    (isfinite(lnom) && isfinite(lnopz_a) && -708<lnom<708 && lnopz_a<708) ||
        error("PRE_REHEATING abundance outside the float64 normal range (ln Omega = $lnom, ln(1+z_a) = $lnopz_a) for log10 m = $logm_ev eV, log10 f = $logf_gev GeV, w_R = $w, T_reh = $treh GeV; it is not clamped to 0 or to a bound. Use a w_R closer to 0 or a higher T_reh (METHODOLOGY.md, numerical range of the PRE prescription)")
    (;omega=exp(lnom),a_osc=exp(-lnopz_a),H_osc=m/3,HR)
end
const EXCL_REASON="m_a>3H_I_max(Fuzzy_fn16;1.4e14_GeV)"
const THETA_REF_NOTE="theta_ref = pi/2 is a normalisation/reference angle only, NOT an adopted, preferred or allowed physical misalignment angle; "*
    "Omega(theta) = Omega_at_theta_ref (theta/theta_ref)^2 holds only inside the small-angle (harmonic) domain, which is established separately"
VAL["inflation_cut"]["enabled"]==true && VAL["mKK_flag"]["exclusion"]==false && VAL["fpert_flag"]["exclusion"]==false &&
    VAL["light_window_100eV"]["applied"]==false || error("config validity switches differ from the approved methodology")
THREE_HI_MAX_GEV==1.4e14 || error("3 H_I,max differs from the paper's stated value")
CFG["axion"]["f_relic_over_f_pert"]==1.0 || error("f_relic must equal f_pert (C = 1)")
CFG["categories_in_cdm_sum"]["FROZEN"]["in_cdm_sum"]==false && CFG["categories_in_cdm_sum"]["LATE"]["in_cdm_sum"]==false &&
    CFG["categories_in_cdm_sum"]["POST_REHEATING"]["in_cdm_sum"]==true && CFG["categories_in_cdm_sum"]["PRE_REHEATING"]["in_cdm_sum"]==true ||
    error("category treatment differs from the approved methodology")

constcheck=OD()
for (name,script,cfgv,tol) in (("H0_eV",H0_GEV*1e9,COS["H0_eV"],1e-15),("omega_m",OMEGA_M_H2,COS["omega_m"],1e-15),
        ("T_CMB_GeV",T_CMB_GEV,COS["T_CMB_GeV"],0.0),("M_pl_reduced_GeV",MPL_REDUCED_GEV,COS["M_pl_reduced_GeV"],0.0),
        ("hbar_GeV_s",HBAR_GEV_S,COS["hbar_GeV_s"],0.0),("H100_GeV",H100_GEV,COS["derived"]["H100_GeV"],1e-15),
        ("rho_crit_over_h2_GeV4",RHO_CRIT_OVER_H2_GEV4,COS["derived"]["rho_crit_over_h2_GeV4"],2e-15),
        ("theta_reference",THETA_INITIAL,CFG["axion"]["theta_reference"],0.0),("w_R",W_REHEAT,get(CFG["reheating"],"w_R",get(CFG["reheating"],"w_R_default",NaN)),0.0),
        ("T_search_max_GeV",T_SEARCH_MAX_GEV,CFG["oscillation"]["T_search_max_GeV"],0.0))
    e=relerr(Float64(script),Float64(cfgv)); e<=tol || error("constant $name: script $script vs config $cfgv")
    push!(constcheck,name=>OD(["script"=>Float64(script),"config"=>Float64(cfgv),"rel_diff"=>e]))
end
const GSTAR=String(COS["gstar_table"])
filesha(GSTAR)==COS["gstar_table_sha256"] || error("g* table checksum mismatch")
const tab=thermal_table(GSTAR)
const teq=equality_temperature(tab); const aeq=scale_factor(tab,teq)
const H0NOW=Hfull(tab,T_CMB_GEV); const HEQ=Hfull(tab,teq)
D=COS["derived"]; dtol=COS["derived_tolerance_rel"]
derivedcheck=OD()
for (name,v) in (("h",HUBBLE_h),("Omega_m0",OMEGA_M0),("Omega_r0",omega_r0(tab)),("Omega_L0",1-OMEGA_M0-omega_r0(tab)),
        ("H_today_GeV",H0NOW),("T_eq_GeV",teq),("three_H_Teq_eV",3*HEQ*1e9))
    tol=Float64(get(dtol,name,dtol["default"])); name=="three_H_Teq_eV" && (tol=Float64(dtol["T_eq_GeV"]))
    e=relerr(v,Float64(D[name])); e<=tol || error("derived $name: julia $v vs config $(D[name]) (tol $tol)")
    push!(derivedcheck,name=>OD(["julia"=>v,"config_reference_B"=>Float64(D[name]),"rel_diff"=>e,"tol"=>tol]))
end
relerr(H0NOW,H0_GEV)<=1e-15 || error("H(today) != H0")
scen_derived=OD()
for (lab,treh,s) in SCEN
    treh>teq || error("T_reh must exceed T_eq")
    HR=hubble_reheat(tab,treh); opzR=onepz_reheat(tab,treh)
    e1=relerr(3HR*1e9,Float64(s["three_H_R_eV"])); e2=relerr(opzR,Float64(s["one_plus_z_R"]))
    max(e1,e2)<=1e-12 || error("scenario $lab: 3H_R or 1+z_R disagree with config")
    HR<THREE_HI_MAX_GEV/3 || error("scenario $lab: H_R >= H_I,max")
    push!(scen_derived,lab=>OD(["T_reh_GeV"=>treh,"H_R_GeV"=>HR,"three_H_R_eV"=>3HR*1e9,"one_plus_z_R"=>opzR,
        "rel_diff_three_H_R_vs_config"=>e1,"rel_diff_one_plus_z_R_vs_config"=>e2]))
    METHOD_VERSION>=2 && (push!(scen_derived[end].second,"w_R"=>WR_OF[lab]))   # [G7]
end

# ---------------------------------------------------------------- input mode table (A1 schema)
lines=readlines(TABLE_PATH); hdr=split(lines[1],','); col=Dict(String(h)=>i for (i,h) in enumerate(hdr))
need=("geometry_id","h11","mode_index","mode_rank_desc","log10_mass_eV","log10_f_pert_GeV","f_relic_over_f_pert",
      "theta_reference","vacuum_status","cy_volume","inflation_valid","above_mKK","above_fpert")
for c in need; haskey(col,c) || error("mode table lacks column $c"); end
modes=NamedTuple[]
for l in lines[2:end]
    isempty(strip(l)) && continue
    x=split(l,','); length(x)==length(hdr) || error("ragged row: $l")
    g(c)=String(x[col[c]])
    push!(modes,(geometry_id=g("geometry_id"),h11=parse(Int,g("h11")),mode_index=parse(Int,g("mode_index")),
        mode_rank_desc=parse(Int,g("mode_rank_desc")),logm=parse(Float64,g("log10_mass_eV")),logf=parse(Float64,g("log10_f_pert_GeV")),
        frel=parse(Float64,g("f_relic_over_f_pert")),theta=parse(Float64,g("theta_reference")),vacuum_status=g("vacuum_status"),
        vol=parse(Float64,g("cy_volume")),a1_valid=parsebool(g("inflation_valid")),a1_mKK=parsebool(g("above_mKK")),
        a1_fpert=parsebool(g("above_fpert"))))
end
const N=length(modes)
length(unique((r.geometry_id,r.mode_index) for r in modes))==N || error("duplicate (geometry_id, mode_index)")
all(r->r.frel==1.0 && r.theta==THETA_INITIAL && r.vacuum_status in ("ORIGINAL_SPECTRUM","VACUUM_CORRECTED","SYNTHETIC_BENCHMARK"),modes) ||
    error("mode table violates f_relic = f_pert, theta = pi/2 or vacuum_status")
all(r->isfinite(r.logm) && isfinite(r.logf) && isfinite(r.vol) && r.vol>0,modes) || error("non-finite input")
is_population=haskey(RUN,"expected_modes") && N==RUN["expected_modes"]   # [G1] expected counts optional
if is_population
    length(unique(r.geometry_id for r in modes))==RUN["expected_geometries"] || error("population geometry count")
    extrema(r.h11 for r in modes)==(RUN["h11_min"],RUN["h11_max"]) || error("population h11 range")
    any(r->r.vacuum_status=="SYNTHETIC_BENCHMARK",modes) && error("synthetic modes in the population table")
    filesha(TABLE_PATH)==CFG["input"]["canonical_mode_table_sha256"] || error("population table is not the pinned A1 canonical table")
end
# [G2] general run on the A1 canonical table: its checksum is always verified (not only for the reference population), and the
# geometry population comes from the A1 canonical geometry table, so a geometry without mode above H0 is never dropped
const CANON=haskey(CFG["input"],"canonical_mode_table") && abspath(String(CFG["input"]["canonical_mode_table"]))==TABLE_PATH
CANON && (filesha(TABLE_PATH)==CFG["input"]["canonical_mode_table_sha256"] ||
    error("canonical mode table checksum differs from run_config.json (the table changed after A1); rerun the relic stage"))
function csvsplit(l)   # comma-separated line with optional double quotes (pandas)
    out=String[]; buf=IOBuffer(); q=false; i=1; c=collect(l)
    while i<=length(c)
        ch=c[i]
        if q && ch=='"' && i<length(c) && c[i+1]=='"'; write(buf,'"'); i+=1
        elseif ch=='"'; q=!q
        elseif ch==',' && !q; push!(out,String(take!(buf)))
        else write(buf,ch) end
        i+=1
    end
    push!(out,String(take!(buf))); out
end
function load_population(path)
    L=readlines(path); h=csvsplit(L[1]); ix=Dict(String(k)=>i for (i,k) in enumerate(h))
    pop=NamedTuple[]
    for l in L[2:end]
        isempty(strip(l)) && continue
        x=csvsplit(l)
        push!(pop,(geometry_id=x[ix["geometry_id"]],h11=parse(Int,x[ix["h11"]]),vacuum_status=x[ix["vacuum_status"]],
            n_modes=parse(Int,x[ix["n_modes"]]),vol=parse(Float64,x[ix["cy_volume"]])))
    end
    pop
end
const POP=if CANON && haskey(CFG["input"],"canonical_geometry_table")
    gp=String(CFG["input"]["canonical_geometry_table"])
    haskey(CFG["input"],"canonical_geometry_table_sha256") && (filesha(gp)==CFG["input"]["canonical_geometry_table_sha256"] ||
        error("canonical geometry table checksum differs from run_config.json"))
    load_population(gp)
else nothing end
if POP!==nothing
    pg=Dict(p.geometry_id=>p for p in POP); length(pg)==length(POP) || error("duplicate geometry in the canonical geometry table")
    pcnt=Dict{String,Int}(); for r in modes; haskey(pg,r.geometry_id) || error("mode of unknown geometry $(r.geometry_id)"); pcnt[r.geometry_id]=get(pcnt,r.geometry_id,0)+1; end
    all(get(pcnt,p.geometry_id,0)==p.n_modes for p in POP) || error("mode counts differ from the canonical geometry table")
    all(r->r.vacuum_status==pg[r.geometry_id].vacuum_status && r.vol==pg[r.geometry_id].vol,modes) || error("geometry fields differ from the canonical geometry table")
end
const PG=POP===nothing ? nothing : Dict(p.geometry_id=>p for p in POP)
if MODE=="abundance" && N>RUN["max_rows_abundance_without_approval"] && RUN["production_run_approved"]!==true
    error("REFUSED: abundance run on $N modes requires run.production_run_approved = true in the configuration (user approval pending)")
end
mkpath(OUTDIR)
println("[$(stamp())] mode=$MODE rows=$N T_eq=$teq GeV H(today)=$(H0NOW*1e9) eV; constants and derived values match the configuration")

# ---------------------------------------------------------------- per-mode evaluation
const CATS=("FROZEN","LATE","POST_REHEATING","PRE_REHEATING")
const STATES=("INCLUDED_POST_REHEATING","INCLUDED_PRE_REHEATING","NOT_INCLUDED_FROZEN","NOT_INCLUDED_LATE","INFLATIONARY_VALIDITY_EXCLUDED")
function classify(m,HR)   # identical to the audited branch order: frozen (solve_tosc NaN), then [F4], then T_osc vs T_eq
    m/3<=H0NOW && return "FROZEN"
    m>3HR && return "PRE_REHEATING"
    m/3<=HEQ ? "LATE" : "POST_REHEATING"
end
function flags(r)
    m=10.0^(r.logm-9); f=10.0^r.logf; mKK=MPL_KK_GEV/r.vol^(2/3)
    valid=!(m>THREE_HI_MAX_GEV)
    valid==r.a1_valid || error("inflation flag disagrees with A1 for $(r.geometry_id) mode $(r.mode_index)")
    (m>mKK)==r.a1_mKK || error("m_KK flag disagrees with A1 for $(r.geometry_id) mode $(r.mode_index)")
    (m>f)==r.a1_fpert || error("f_pert flag disagrees with A1 for $(r.geometry_id) mode $(r.mode_index)")
    (;m,f,mKK,valid,above_mKK=m>mKK,above_fpert=m>f)
end
state_of(cat,valid)=cat=="FROZEN" ? "NOT_INCLUDED_FROZEN" : !valid ? "INFLATIONARY_VALIDITY_EXCLUDED" :
    cat=="LATE" ? "NOT_INCLUDED_LATE" : cat=="POST_REHEATING" ? "INCLUDED_POST_REHEATING" : "INCLUDED_PRE_REHEATING"
function evaluate(r,lab,treh,HR)
    fl=flags(r); m=fl.m; cat=classify(m,HR); st=state_of(cat,fl.valid)
    Tosc=nothing; z=nothing; Hosc=nothing; resid=nothing; om=nothing; omlate=nothing
    if st=="NOT_INCLUDED_FROZEN"; branch="frozen_not_oscillating"
    elseif st=="INFLATIONARY_VALIDITY_EXCLUDED"; branch="not_evaluated_inflation_invalid"
    elseif cat=="PRE_REHEATING"
        pr=try pre_reheating_abundance_w(tab,r.logm,r.logf,treh,WR_OF[lab]) catch e   # (2.10)+(2.15); [G7] w_R of the scenario (0 in v1)
            error("scenario $lab, geometry $(r.geometry_id), mode $(r.mode_index): "*(e isa ErrorException ? e.msg : sprint(showerror,e)))
        end
        branch=WR_OF[lab]==0.0 ? "fuzzy_wR0_eq2.10_2.15" : "fuzzy_wR_general_eq2.10_2.15"; z=1/pr.a_osc-1; Hosc=pr.H_osc; om=pr.omega
    else
        tosc=solve_tosc(tab,m)
        isfinite(tosc) && tosc>T_CMB_GEV || error("T_osc not found for $(r.geometry_id) mode $(r.mode_index)")
        ((tosc<=teq)==(cat=="LATE") || abs(log(tosc/teq))<=1e-10) ||   # [G3] class from the exact threshold; T_osc and T_eq carry |dlnT|<1e-11
            error("threshold class $cat disagrees with solved T_osc for $(r.geometry_id) mode $(r.mode_index)")
        # T_osc carries the audited bisection tolerance |log(hi/lo)| < 1e-11; the class itself comes from the exact thresholds
        tosc<=treh*(1+1e-10) || error("post-reheating onset above T_reh beyond the root tolerance")
        raw=raw_abundance(tab,r.logm,r.logf,tosc)
        branch=cat=="LATE" ? "late_full_H_census_only" : "post_reheating_full_H"
        Tosc=tosc; z=1/scale_factor(tab,tosc)-1; Hosc=m/3; resid=raw.residual
        cat=="LATE" ? (omlate=raw.omega) : (om=raw.omega)
    end
    om===nothing || (isfinite(om) && om>0) || error("invalid abundance $om")
    Dict{String,Any}("geometry_id"=>r.geometry_id,"h11"=>r.h11,"mode_index"=>r.mode_index,"mode_rank_desc"=>r.mode_rank_desc,
        "mass_eV"=>10.0^r.logm,"log10_mass_eV"=>r.logm,"f_pert_GeV"=>fl.f,"log10_f_pert_GeV"=>r.logf,"theta_ref"=>THETA_INITIAL,
        "T_reh_GeV"=>treh,"scenario"=>lab,"category"=>cat,"abundance_state"=>st,"abundance_branch"=>branch,
        "T_osc_GeV"=>Tosc,"z_osc"=>z,"H_osc_GeV"=>Hosc,"threeH_over_m_minus_one"=>resid,
        "Omega_h2_at_theta_ref"=>om,"included_in_cdm_sum"=>startswith(st,"INCLUDED_"),"Omega_h2_late_census_only_at_theta_ref"=>omlate,
        "inflation_valid"=>fl.valid,"inflation_exclusion_reason"=>fl.valid ? nothing : EXCL_REASON,
        "mass_over_3HImax"=>m/THREE_HI_MAX_GEV,"m_KK_GeV"=>fl.mKK,"above_mKK"=>fl.above_mKK,"mass_over_mKK"=>m/fl.mKK,
        "above_fpert"=>fl.above_fpert,"mass_over_fpert"=>m/fl.f,"vacuum_status"=>r.vacuum_status,"cy_volume"=>r.vol)
end
const MASTER_COLS=["geometry_id","h11","mode_index","mode_rank_desc","mass_eV","log10_mass_eV","f_pert_GeV","log10_f_pert_GeV",
    "theta_ref","T_reh_GeV","scenario","category","abundance_state","abundance_branch","T_osc_GeV","z_osc","H_osc_GeV",
    "threeH_over_m_minus_one","Omega_h2_at_theta_ref","included_in_cdm_sum","Omega_h2_late_census_only_at_theta_ref","inflation_valid",
    "inflation_exclusion_reason","mass_over_3HImax","m_KK_GeV","above_mKK","mass_over_mKK","above_fpert","mass_over_fpert","vacuum_status"]
const GEOM_COLS=["geometry_id","h11","vacuum_status","scenario","T_reh_GeV","theta_ref","number_of_modes","number_frozen","number_late",
    "number_post_reheating","number_pre_reheating","number_pre_reheating_valid","number_inflation_excluded","number_included",
    "Omega_geometry_h2_at_theta_ref","Omega_post_reheating_h2_at_theta_ref","Omega_pre_reheating_h2_at_theta_ref","omega_total_excludes_invalid_modes",
    "dominant_mode_index","dominant_fraction","Omega_late_census_only_h2_at_theta_ref","number_above_mKK","number_above_fpert",
    "number_included_above_mKK","number_included_above_fpert","Omega_included_above_mKK_h2_at_theta_ref","Omega_included_above_fpert_h2_at_theta_ref",
    "cy_volume","m_KK_GeV"]
sumornothing(v)=isempty(v) ? nothing : sum(v)
function geometry_rows(rows,lab,treh,with_omega=true)
    order=String[]; idx=Dict{String,Vector{Int}}()
    for (k,r) in enumerate(rows); haskey(idx,r["geometry_id"]) || push!(order,r["geometry_id"]); push!(get!(idx,r["geometry_id"],Int[]),k); end
    POP===nothing || (order=[p.geometry_id for p in POP])   # [G4] every A1 geometry, in A1 order (the same order as the mode table)
    out=Dict{String,Any}[]
    for gid in order
        if !haskey(idx,gid)                                  # [G4] validated spectrum without mode above H0: evaluated, empty
            p=PG[gid]; mKK=MPL_KK_GEV/p.vol^(2/3)
            push!(out,Dict{String,Any}("geometry_id"=>gid,"h11"=>p.h11,"vacuum_status"=>p.vacuum_status,"scenario"=>lab,"T_reh_GeV"=>treh,
                "theta_ref"=>THETA_INITIAL,"number_of_modes"=>0,"number_frozen"=>0,"number_late"=>0,"number_post_reheating"=>0,
                "number_pre_reheating"=>0,"number_pre_reheating_valid"=>0,"number_inflation_excluded"=>0,"number_included"=>0,
                "Omega_geometry_h2_at_theta_ref"=>with_omega ? 0.0 : nothing,"Omega_post_reheating_h2_at_theta_ref"=>nothing,
                "Omega_pre_reheating_h2_at_theta_ref"=>nothing,"omega_total_excludes_invalid_modes"=>false,"dominant_mode_index"=>nothing,
                "dominant_fraction"=>nothing,"Omega_late_census_only_h2_at_theta_ref"=>nothing,"number_above_mKK"=>0,"number_above_fpert"=>0,
                "number_included_above_mKK"=>0,"number_included_above_fpert"=>0,"Omega_included_above_mKK_h2_at_theta_ref"=>nothing,
                "Omega_included_above_fpert_h2_at_theta_ref"=>nothing,"cy_volume"=>p.vol,"m_KK_GeV"=>mKK))
            continue
        end
        R=rows[idx[gid]]; cnt(f)=count(f,R)
        inc=filter(r->r["included_in_cdm_sum"],R)
        oms=with_omega ? Float64[r["Omega_h2_at_theta_ref"] for r in inc] : Float64[]
        nexc=cnt(r->r["abundance_state"]=="INFLATIONARY_VALIDITY_EXCLUDED")
        tot=with_omega ? (!isempty(oms) ? sum(oms) : nexc==0 ? 0.0 : nothing) : nothing   # [G4] empty CDM set: 0 if every mode was evaluated, NA if a mode is not evaluated (inflation cut)
        dom=isempty(oms) ? nothing : inc[argmax(oms)]
        som(v)=with_omega ? sumornothing(v) : nothing
        length(unique(r["vacuum_status"] for r in R))==1 && length(unique(r["cy_volume"] for r in R))==1 || error("geometry fields differ within $gid")
        push!(out,Dict{String,Any}("geometry_id"=>gid,"h11"=>R[1]["h11"],"vacuum_status"=>R[1]["vacuum_status"],"scenario"=>lab,
            "T_reh_GeV"=>treh,"theta_ref"=>THETA_INITIAL,"number_of_modes"=>length(R),
            "number_frozen"=>cnt(r->r["category"]=="FROZEN"),"number_late"=>cnt(r->r["category"]=="LATE"),
            "number_post_reheating"=>cnt(r->r["category"]=="POST_REHEATING"),"number_pre_reheating"=>cnt(r->r["category"]=="PRE_REHEATING"),
            "number_pre_reheating_valid"=>cnt(r->r["category"]=="PRE_REHEATING" && r["inflation_valid"]),
            "number_inflation_excluded"=>cnt(r->r["abundance_state"]=="INFLATIONARY_VALIDITY_EXCLUDED"),
            "number_included"=>length(inc),"Omega_geometry_h2_at_theta_ref"=>tot,
            "Omega_post_reheating_h2_at_theta_ref"=>som([r["Omega_h2_at_theta_ref"] for r in inc if r["category"]=="POST_REHEATING"]),
            "Omega_pre_reheating_h2_at_theta_ref"=>som([r["Omega_h2_at_theta_ref"] for r in inc if r["category"]=="PRE_REHEATING"]),
            "omega_total_excludes_invalid_modes"=>any(r->!r["inflation_valid"],R),
            "dominant_mode_index"=>dom===nothing ? nothing : dom["mode_index"],
            "dominant_fraction"=>dom===nothing ? nothing : dom["Omega_h2_at_theta_ref"]/tot,
            "Omega_late_census_only_h2_at_theta_ref"=>som([r["Omega_h2_late_census_only_at_theta_ref"] for r in R if r["category"]=="LATE"]),
            "number_above_mKK"=>cnt(r->r["above_mKK"]),"number_above_fpert"=>cnt(r->r["above_fpert"]),
            "number_included_above_mKK"=>count(r->r["above_mKK"],inc),"number_included_above_fpert"=>count(r->r["above_fpert"],inc),
            "Omega_included_above_mKK_h2_at_theta_ref"=>som([r["Omega_h2_at_theta_ref"] for r in inc if r["above_mKK"]]),
            "Omega_included_above_fpert_h2_at_theta_ref"=>som([r["Omega_h2_at_theta_ref"] for r in inc if r["above_fpert"]]),
            "cy_volume"=>R[1]["cy_volume"],"m_KK_GeV"=>R[1]["m_KK_GeV"]))
    end
    out
end
function quant(v,q) s=sort(v); s[clamp(round(Int,q*(length(s)-1))+1,1,length(s))] end
function census_block(rows,G)
    byc=OD([c=>count(r->r["category"]==c,rows) for c in CATS]); bys=OD([s=>count(r->r["abundance_state"]==s,rows) for s in STATES])
    exc=filter(r->!r["inflation_valid"],rows)
    cb=OD(["modes"=>length(rows),"geometries"=>length(G),"by_category"=>byc,"category_sum"=>sum(last.(byc)),"by_abundance_state"=>bys,   # [G5]
        "included_in_cdm_sum"=>count(r->r["included_in_cdm_sum"],rows),
        "inflation_excluded_modes"=>length(exc),"inflation_excluded_geometries"=>length(unique(r["geometry_id"] for r in exc)),
        "inflation_excluded_by_category"=>OD([c=>count(r->r["category"]==c,exc) for c in CATS]),
        "above_mKK_modes"=>count(r->r["above_mKK"],rows),"above_mKK_included"=>count(r->r["above_mKK"] && r["included_in_cdm_sum"],rows),
        "above_fpert_modes"=>count(r->r["above_fpert"],rows),"above_fpert_included"=>count(r->r["above_fpert"] && r["included_in_cdm_sum"],rows),
        "geometries_with_included_modes"=>count(g->g["number_included"]>0,G),
        "geometries_without_included_modes"=>count(g->g["number_included"]==0,G),
        "geometries_without_included_modes_all_frozen_or_late"=>count(g->g["number_included"]==0 && g["number_inflation_excluded"]==0 && g["number_of_modes"]>0,G),   # [G5]
        "geometries_without_included_modes_due_to_inflation_cut"=>count(g->g["number_included"]==0 && g["number_inflation_excluded"]>0,G),
        "geometries_with_inflation_excluded_modes"=>count(g->g["number_inflation_excluded"]>0,G),
        "geometries_partially_excluded_keep_valid_modes"=>count(g->g["number_inflation_excluded"]>0 && g["number_inflation_excluded"]<g["number_of_modes"],G)])
    nz=count(g->g["number_of_modes"]==0,G); nz>0 && push!(cb,"geometries_without_modes_above_threshold"=>nz)   # [G5] key present only when such geometries exist
    cb
end
# [G6] abundance validity of every geometry (abundance runs; approved 2026-10-31):
#   EVALUATED_POSITIVE     no mode excluded, >= 1 mode in the CDM sum: complete abundance > 0          -> theta PRIMARY
#   EVALUATED_ZERO         no mode excluded, none in the CDM sum (no mode above H0, or all FROZEN/LATE):
#                          complete abundance = 0 (evaluated empty sum)                                 -> theta PRIMARY
#   PARTIAL_VALIDITY       some, not all, modes above 3H_I,max (abundance not evaluated for them): only a
#                          partial sum over the valid modes, not the complete abundance                 -> theta SENSITIVITY_ONLY
#   INFLATION_INVALID_ALL  every mode above 3H_I,max: abundance not evaluated (NA)                     -> theta EXCLUDED_NOT_EVALUATED
const VALIDITY_COLS=["geometry_id","h11","scenario","number_of_modes","number_included","number_inflation_excluded","abundance_validity","reason",
    "Omega_complete_h2_at_theta_ref","Omega_valid_modes_partial_h2_at_theta_ref","theta_population"]
function validity_rows(G)
    map(G) do g
        n=g["number_of_modes"]; ni=g["number_included"]; fl=g["number_frozen"]+g["number_late"]; ex=g["number_inflation_excluded"]; om=g["Omega_geometry_h2_at_theta_ref"]
        cls=ex==0 ? (ni>0 ? "EVALUATED_POSITIVE" : "EVALUATED_ZERO") : ex==n ? "INFLATION_INVALID_ALL" : "PARTIAL_VALIDITY"
        reason=cls=="EVALUATED_POSITIVE" ? nothing : cls=="EVALUATED_ZERO" ? (n==0 ? "NO_MODES_ABOVE_THRESHOLD" : "ALL_MODES_FROZEN_OR_LATE") :
            cls=="PARTIAL_VALIDITY" ? "SOME_MODES_INFLATION_EXCLUDED" : "ALL_MODES_INFLATION_EXCLUDED"
        Dict{String,Any}("geometry_id"=>g["geometry_id"],"h11"=>g["h11"],"scenario"=>g["scenario"],"number_of_modes"=>n,"number_included"=>ni,
            "number_inflation_excluded"=>ex,"abundance_validity"=>cls,"reason"=>reason,
            "Omega_complete_h2_at_theta_ref"=>startswith(cls,"EVALUATED_") ? om : nothing,
            "Omega_valid_modes_partial_h2_at_theta_ref"=>cls=="PARTIAL_VALIDITY" ? (om===nothing ? 0.0 : om) : nothing,
            "theta_population"=>startswith(cls,"EVALUATED_") ? "PRIMARY" : cls=="PARTIAL_VALIDITY" ? "SENSITIVITY_ONLY" : "EXCLUDED_NOT_EVALUATED")
    end
end
validity_counts(V)=OD([c=>count(v->v["abundance_validity"]==c,V) for c in ("EVALUATED_POSITIVE","EVALUATED_ZERO","PARTIAL_VALIDITY","INFLATION_INVALID_ALL")])
function h11_rows(G,lab,with_omega)
    out=Dict{String,Any}[]
    for h in sort(unique(g["h11"] for g in G))
        Gh=filter(g->g["h11"]==h,G); inc=filter(g->g["number_included"]>0,Gh)
        push!(out,Dict{String,Any}("h11"=>h,"scenario"=>lab,"geometries"=>length(Gh),"modes"=>sum(g["number_of_modes"] for g in Gh),
            "n_FROZEN"=>sum(g["number_frozen"] for g in Gh),"n_LATE"=>sum(g["number_late"] for g in Gh),
            "n_POST_REHEATING"=>sum(g["number_post_reheating"] for g in Gh),"n_PRE_REHEATING"=>sum(g["number_pre_reheating"] for g in Gh),
            "n_inflation_excluded"=>sum(g["number_inflation_excluded"] for g in Gh),"n_included"=>sum(g["number_included"] for g in Gh),
            "geometries_with_included"=>length(inc),
            "p16_log10_Omega_geometry_h2_at_theta_ref"=>(with_omega && !isempty(inc)) ? quant([log10(g["Omega_geometry_h2_at_theta_ref"]) for g in inc],0.16) : nothing,
            "median_log10_Omega_geometry_h2_at_theta_ref"=>(with_omega && !isempty(inc)) ? quant([log10(g["Omega_geometry_h2_at_theta_ref"]) for g in inc],0.5) : nothing,
            "p84_log10_Omega_geometry_h2_at_theta_ref"=>(with_omega && !isempty(inc)) ? quant([log10(g["Omega_geometry_h2_at_theta_ref"]) for g in inc],0.84) : nothing))
    end
    out
end
const H11_COLS=["h11","scenario","geometries","modes","n_FROZEN","n_LATE","n_POST_REHEATING","n_PRE_REHEATING","n_inflation_excluded",
    "n_included","geometries_with_included","p16_log10_Omega_geometry_h2_at_theta_ref","median_log10_Omega_geometry_h2_at_theta_ref",
    "p84_log10_Omega_geometry_h2_at_theta_ref"]

# ---------------------------------------------------------------- run
# [G7] v2: column w_R appended to every table (schema relic_abundance_production_outputs/2); v1 tables are unchanged
tagw(X,lab)=(for x in X; x["w_R"]=WR_OF[lab]; end; X)
METHOD_VERSION>=2 && for c in (MASTER_COLS,GEOM_COLS,VALIDITY_COLS,H11_COLS); push!(c,"w_R"); end
summary=OD(); POPSUM=OD(); census_cols=["geometry_id","h11","mode_index","mode_rank_desc","log10_mass_eV","vacuum_status","inflation_valid","above_mKK","above_fpert"]   # [G6]
census_rows=[Dict{String,Any}("geometry_id"=>r.geometry_id,"h11"=>r.h11,"mode_index"=>r.mode_index,"mode_rank_desc"=>r.mode_rank_desc,
    "log10_mass_eV"=>r.logm,"vacuum_status"=>r.vacuum_status) for r in modes]
for (lab,treh,s) in SCEN
    HR=hubble_reheat(tab,treh)
    if MODE=="census"
        rows=Dict{String,Any}[]
        for r in modes
            fl=flags(r); cat=classify(fl.m,HR); st=state_of(cat,fl.valid)
            push!(rows,Dict{String,Any}("geometry_id"=>r.geometry_id,"h11"=>r.h11,"mode_index"=>r.mode_index,"category"=>cat,
                "abundance_state"=>st,"included_in_cdm_sum"=>startswith(st,"INCLUDED_"),"inflation_valid"=>fl.valid,
                "above_mKK"=>fl.above_mKK,"above_fpert"=>fl.above_fpert,"vacuum_status"=>r.vacuum_status,"cy_volume"=>r.vol,"m_KK_GeV"=>fl.mKK,
                "Omega_h2_at_theta_ref"=>nothing,"Omega_h2_late_census_only_at_theta_ref"=>nothing,"mass_over_3HImax"=>fl.m/THREE_HI_MAX_GEV))
        end
        for (k,r) in enumerate(rows)
            c=census_rows[k]; c["inflation_valid"]=r["inflation_valid"]; c["above_mKK"]=r["above_mKK"]; c["above_fpert"]=r["above_fpert"]
            c["category_$lab"]=r["category"]; c["abundance_state_$lab"]=r["abundance_state"]; c["included_in_cdm_sum_$lab"]=r["included_in_cdm_sum"]
        end
        append!(census_cols,["category_$lab","abundance_state_$lab","included_in_cdm_sum_$lab"])
        G=geometry_rows(rows,lab,treh,false)
        write_table(joinpath(OUTDIR,"census_h11_$lab.csv"),H11_COLS,tagw(h11_rows(G,lab,false),lab))
        push!(summary,lab=>OD(["T_reh_GeV"=>treh,"three_H_R_eV"=>3HR*1e9,"census"=>census_block(rows,G),"abundance"=>"NOT COMPUTED (census/dry-run mode)"]))
        METHOD_VERSION>=2 && (push!(summary[end].second,"w_R"=>WR_OF[lab]))   # [G7]
    else
        rows=tagw([evaluate(r,lab,treh,HR) for r in modes],lab)
        G=tagw(geometry_rows(rows,lab,treh),lab)
        write_table(joinpath(OUTDIR,"master_modes_$lab.csv"),MASTER_COLS,rows)
        write_table(joinpath(OUTDIR,"geometry_summary_$lab.csv"),GEOM_COLS,G)
        write_table(joinpath(OUTDIR,"h11_summary_$lab.csv"),H11_COLS,tagw(h11_rows(G,lab,true),lab))
        VR=tagw(validity_rows(G),lab); write_table(joinpath(OUTDIR,"geometry_abundance_validity_$lab.csv"),VALIDITY_COLS,VR)   # [G6]
        push!(POPSUM,lab=>OD(["geometries"=>length(VR),"by_abundance_validity"=>validity_counts(VR),   # [G6]
            "theta_primary_geometries"=>count(v->v["theta_population"]=="PRIMARY",VR),"theta_sensitivity_geometries"=>count(v->v["theta_population"]!="EXCLUDED_NOT_EVALUATED",VR),
            "theta_excluded_not_evaluated"=>count(v->v["theta_population"]=="EXCLUDED_NOT_EVALUATED",VR)]))
        inc=filter(g->g["number_included"]>0,G); tots=[g["Omega_geometry_h2_at_theta_ref"] for g in inc]
        pre=[r["Omega_h2_at_theta_ref"] for r in rows if r["abundance_state"]=="INCLUDED_PRE_REHEATING"]
        ab=isempty(tots) ? OD() : OD(["geometries_with_included"=>length(tots),
            "Omega_geometry_h2_at_theta_ref_quantiles"=>OD([string(q)=>quant(tots,q) for q in (0.0,0.05,0.16,0.5,0.84,0.95,1.0)]),
            "theta_ref"=>THETA_INITIAL,"theta_ref_interpretation"=>THETA_REF_NOTE,
            "sum_over_included_modes_POST"=>sumornothing([r["Omega_h2_at_theta_ref"] for r in rows if r["abundance_state"]=="INCLUDED_POST_REHEATING"]),
            "sum_over_included_modes_PRE"=>sumornothing(pre),"note"=>"sums across geometries are bookkeeping only; each geometry is a separate universe"])
        push!(summary,lab=>OD(["T_reh_GeV"=>treh,"three_H_R_eV"=>3HR*1e9,"census"=>census_block(rows,G),"abundance"=>ab]))
        METHOD_VERSION>=2 && (push!(summary[end].second,"w_R"=>WR_OF[lab]))   # [G7]
    end
end
MODE=="census" && write_table(joinpath(OUTDIR,"census_modes.csv"),census_cols,census_rows)
MODE=="abundance" && json_write(joinpath(OUTDIR,"population_validity_summary.json"),OD(["definition"=>"abundance validity classes (METHODOLOGY.md); theta primary = EVALUATED_POSITIVE + EVALUATED_ZERO, sensitivity adds PARTIAL_VALIDITY (partial sums), INFLATION_INVALID_ALL is not evaluated",
    "scenarios"=>POPSUM]))   # [G6]
json_write(joinpath(OUTDIR,MODE=="census" ? "census_summary.json" : "global_summary.json"),summary)
meta=OD(["mode"=>MODE,"rows"=>N,"is_full_population"=>is_population,"production_run_approved"=>RUN["production_run_approved"],
    "script"=>abspath(@__FILE__),"script_sha256"=>filesha(abspath(@__FILE__)),"config"=>CONFIG_PATH,"config_sha256"=>filesha(CONFIG_PATH),
    "mode_table"=>TABLE_PATH,"mode_table_sha256"=>filesha(TABLE_PATH),"gstar_table_sha256"=>filesha(GSTAR),
    "manifest_sha256"=>CFG["input"]["manifest_sha256"],"julia_version"=>string(VERSION),
    "constants_checked_against_config"=>constcheck,"derived_checked_against_config"=>derivedcheck,"scenarios"=>scen_derived,
    "cosmology"=>OD(["H0_eV"=>H0_GEV*1e9,"H_today_eV"=>H0NOW*1e9,"h"=>HUBBLE_h,"omega_m"=>OMEGA_M_H2,"Omega_m0"=>OMEGA_M0,
        "Omega_r0"=>omega_r0(tab),"Omega_L0"=>1-OMEGA_M0-omega_r0(tab),"T_eq_GeV"=>teq,"T_eq_eV"=>teq*1e9,"a_eq"=>aeq,
        "three_H0_eV"=>3*H0NOW*1e9,"three_H_Teq_eV"=>3*HEQ*1e9,"M_pl_reduced_GeV"=>MPL_REDUCED_GEV]),
    "methodology"=>OD(["criterion"=>"m_a = 3H","theta_ref"=>THETA_INITIAL,"theta_ref_interpretation"=>THETA_REF_NOTE,
        "theta_rescaling"=>"Omega(theta) = Omega_h2_at_theta_ref * (theta/theta_ref)^2, harmonic; valid only in the small-angle domain (not pi/2 by default)",
        "theta_scan_performed"=>false,"f_relic"=>"f_pert (C = 1)","w_R"=>(METHOD_VERSION==1 ? W_REHEAT : "per scenario: see scenarios[].w_R"),
        "pre_reheating"=>"Fuzzy (2.10)+(2.15); printed (2.16) normalisation not used","three_H_I_max_GeV"=>THREE_HI_MAX_GEV,
        "inflation_cut_unit"=>"mode","mKK_flag_M_pl_GeV"=>MPL_KK_GEV,"mKK_and_fpert"=>"flags only","light_window_100eV"=>"not applied",
        "FROZEN_LATE"=>"census only"])])
METHOD_VERSION>=2 && pushfirst!(meta,"methodology_version"=>METHOD_VERSION)   # [G7] v1 metadata keeps its frozen key set
json_write(joinpath(OUTDIR,"run_metadata.json"),meta)
println("[$(stamp())] done: outputs in $OUTDIR")
