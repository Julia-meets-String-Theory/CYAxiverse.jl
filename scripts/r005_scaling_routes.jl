# CYAX-0232 R-005: the c = 1/2 comparison harness (T013).
#
# `r005_route_outputs(; kinv_scale)` evaluates every R-005 route on the R-004
# and R-003 fixture inputs with the stored inverse metric `kinv_scale · Kinv`
# and returns the route outputs keyed "fixture/route/field". Each route reads
# `Kinv` only to build the kinetic factor or matrix, so `kinv_scale = 2`
# evaluates the c = 1/2 convention with the c = 1 code (K = 2 (2 Kinv)⁻¹ =
# Kinv⁻¹), and `kinv_scale = 1` is the package convention.
#
# This file defines functions only. Callers must load CYAxiverse, HDF5 and
# LinearAlgebra and the script functions `run_batch`
# (scripts/batch_physical_spectrum.jl), `run_geometry`,
# `write_inflation_group!` (scripts/build_orientifold_vacua_inflation.jl);
# test/runtests.jl and scripts/r005_scaling_harness.jl do so.
#
# Fixture inputs are pinned in specs/0232-kinetic-normalization/plan.md
# (*Analytic fixture (R-004)* and *R-005 row map*); the R-003 inputs are the
# literals of the "Fuzzy-axion h11=2 source-corrected role validation" test.

const R005_GS = 0.5029733
const R005_W0 = 1.0

# Eq. 3.18 instanton scales as `leading_axion_reference_data` builds them:
# log10[τ e^{-2πτ}] plus log10(8π sqrt(P) m_{3/2} / V), m_{3/2} in Planck units.
function r005_eq318(tau, cy_volume; gs=R005_GS, w0=R005_W0)
    PB = CYAxiverse.paper_benchmarks
    prefactor = PB.fuzzy_axion_prefactor_P(gs)
    gravitino = PB.fuzzy_axion_gravitino_mass(prefactor,
        PB.fuzzy_axion_kahler_potential(cy_volume), w0; mplanck_ev=1.0)
    L = PB.instanton_scales(Float64.(tau), 1.0)
    L[2, :] .+= log10(8π) + 0.5 * log10(Float64(prefactor)) +
        log10(Float64(gravitino)) - log10(Float64(cy_volume))
    (; L, prefactor, gravitino)
end

"""
    r005_fixtures()

The R-005 fixtures and the routes each one runs:
- `r004`: the R-004 non-diagonal fixture (spectrum, axion–photon, model
  stage). It carries the QED visible-sector metadata of the plan's F3 row:
  QED charge (1, 1) (stored instanton 2, kinetic norm 7 at c = 1/2, 3.5 at
  c = 1), with the two modes at 10^24.26 eV and 10^-1.14 eV on either side of
  the electron mass.
- `r003`: the R-003 source fixture (spectrum, axion–photon, model stage).
- `infl`: the inflation fixture of the plan's row map (runtests.jl:645–704,
  full.jl:784–840 charges with `Kinv = [3 1; 1 2]`), for the in-memory
  diagnostics at non-stationary points, `run_geometry` and F6. The R-004
  fixture is not used there: its leading scales differ by 50 decades, beyond
  the Float64 eigen route of `run_geometry`, whose light mass is then roundoff
  (in both conventions). `oriented_potential` rejects the R-003 charge layout.
"""
function r005_fixtures()
    r004_tau = [1.0, 20.0, 30.0]
    r004 = r005_eq318(r004_tau, 50.0)
    r003_tau = Float64[312.28056480059246, 34.041122071663615, 34.041122071663615,
        68.08224414332723, 139.11972136446443, 36.99635514947357]
    r003_volume = 222.48322433885366
    r003 = r005_eq318(r003_tau, r003_volume)
    (
        (name="r004", h11=2, np=1, Q=Int[1 1 0; 0 1 1], tau=r004_tau,
         cy_volume=50.0, L=r004.L, prefactor=r004.prefactor,
         gravitino=r004.gravitino, Kinv=Float64[3 1; 1 2],
         visible=(qcd=0, qed=1, qed_instanton=1, qed_charge=Int[1, 1]),
         routes=(:spectrum, :axion_photon, :model_stage)),
        (name="r003", h11=2, np=2, Q=Int[7 1 1 2 3 0; 2 0 0 0 1 1], tau=r003_tau,
         cy_volume=r003_volume, L=r003.L, prefactor=r003.prefactor,
         gravitino=r003.gravitino,
         Kinv=Float64[2238.718569786352 -2617.525402950724;
             -2617.525402950724 28440.2666884343],
         visible=nothing, routes=(:spectrum, :axion_photon, :model_stage)),
        (name="infl", h11=2, np=3, Q=Int[1 0 1; 0 1 1], tau=[1.0, 1.0, 1.0],
         cy_volume=10.0, L=Float64[1 1 1; -1 -2 -3], Kinv=Float64[3 1; 1 2],
         visible=nothing, routes=(:spectrum, :inflation),
         theta_points=([0.1, 0.05], [0.23, 0.41], [0.37, 0.12])),
    )
end

function r005_write_geometry(root, f, kinv_scale)
    dir = joinpath(root, "h11_" * lpad(f.h11, 3, '0'),
        "np_" * lpad(f.np, 7, '0'), "cy_0000001")
    mkpath(dir)
    path = joinpath(dir, "cyax.h5")
    N = size(f.Q, 2)
    h5open(path, "w") do file
        cytools = create_group(file, "cytools")
        potential = create_group(cytools, "potential")
        potential["Q"] = f.Q
        potential["L"] = f.L
        geometric = create_group(cytools, "geometric")
        geometric["Kinv"] = kinv_scale .* f.Kinv
        geometric["tip"] = ones(Float64, f.h11)
        geometric["divisor_volumes"] = ones(Float64, f.h11)
        geometric["CY_volume"] = f.cy_volume
        geometric["effective_cone"] = f.Q
        geometric["prime_divisor_volumes"] = f.tau
        geometric["prime_toric_divisors"] = collect(1:N)
        if f.visible !== nothing
            v = f.visible
            visible = create_group(geometric, "visible_sector")
            visible["qcd_divisor_index"] = v.qcd
            visible["qed_divisor_index"] = v.qed
            visible["qcd_image_index"] = v.qcd
            visible["qed_image_index"] = v.qed
            visible["qcd_divisor_volume"] = f.tau[v.qcd + 1]
            visible["qed_divisor_volume"] = f.tau[v.qed + 1]
            visible["qcd_charge"] = f.Q[:, v.qcd + 1]
            visible["qed_charge"] = v.qed_charge
            visible["em_charge"] = v.qed_charge
            visible["qed_instanton_index"] = v.qed_instanton
            visible["qed_log10_lambda4"] = f.L[2, v.qed_instanton + 1]
            visible["qcd_qed_intersection"] = 1
            visible["qcd_invariant"] = 1
            visible["qed_invariant"] = 1
        end
    end
    path
end

function r005_read_group!(out, prefix, group)
    for name in keys(group)
        object = group[name]
        if object isa HDF5.Group
            r005_read_group!(out, prefix * "/" * name, object)
        else
            out[prefix * "/" * name] = read(object)
        end
    end
    out
end

# Run `f(geom_idx, path)` on a fresh copy of fixture `fixture` in its own data
# directory, with the stored inverse metric scaled by `kinv_scale`.
function r005_with_geometry(f, fixture, kinv_scale)
    mktempdir() do root
        path = r005_write_geometry(root, fixture, kinv_scale)
        geom_idx = CYAxiverse.structs.GeometryIndex(fixture.h11, fixture.np, 1)
        withenv("CYAXIVERSE_DATA_DIR" => root) do
            f(geom_idx, path, root)
        end
    end
end

r005_errline(err) = "error:" * string(nameof(typeof(err))) * ":" *
    first(split(sprint(showerror, err), '\n'))

# Record a route's outputs, or its error (type and first message line), so a
# route that fails at the base must fail the same way at the candidate.
function r005_route!(f, out, prefix)
    try
        f()
    catch err
        out[prefix * "/error"] = r005_errline(err)
    end
    out
end

# Numeric fields of an in-memory result (a struct or a Dict).
function r005_record!(out, prefix, x)
    pairs_ = x isa AbstractDict ? ((string(k), v) for (k, v) in x) :
        ((string(k), getfield(x, k)) for k in fieldnames(typeof(x)))
    for (name, value) in pairs_
        if value isa Number || value isa AbstractArray{<:Number}
            out[prefix * "/" * name] = value
        end
    end
    out
end

function r005_spectrum_routes!(out, fixture, kinv_scale)
    key = fixture.name
    phase = fill(0.05, fixture.h11)
    # F2 in memory (no cubic tensor).
    r005_with_geometry(fixture, kinv_scale) do geom_idx, path, _
        r005_route!(out, key * "/F2_hp_memory") do
            r005_record!(out, key * "/F2_hp_memory", CYAxiverse.generate.hp_spectrum(geom_idx))
        end
        r005_route!(out, key * "/F2_pq_memory") do
            r005_record!(out, key * "/F2_pq_memory", CYAxiverse.generate.pq_spectrum(geom_idx))
        end
    end
    # F2 persisted: hp_spectrum_save (ϑ/charge-basis cubic) and pq_spectrum_save.
    r005_with_geometry(fixture, kinv_scale) do geom_idx, path, _
        r005_route!(out, key * "/F2_hp") do
            CYAxiverse.generate.hp_spectrum_save(fixture.h11, fixture.np, 1; phase=phase)
            h5open(path, "r") do file
                r005_read_group!(out, key * "/F2_hp", file["spectrum"])
            end
        end
    end
    r005_with_geometry(fixture, kinv_scale) do geom_idx, path, _
        r005_route!(out, key * "/F2_pq") do
            CYAxiverse.generate.pq_spectrum_save(fixture.h11, fixture.np, 1; phase=phase)
            h5open(path, "r") do file
                r005_read_group!(out, key * "/F2_pq", file["spectrum"])
            end
        end
    end
    # F1: the physical-spectrum batch runner, with quartics.
    r005_with_geometry(fixture, kinv_scale) do geom_idx, path, root
        options = (data_dir=root, h11=nothing, limit=nothing, offset=0,
            geometries=[geom_idx], prec=200, threshold_log10=nothing,
            quartics=true, force=true, summary=joinpath(root, "summary.csv"),
            append_summary=false, hilbert=false)
        redirect_stdout(devnull) do
            run_batch(options) || error("R-005 F1 run_batch failed for $(fixture.name)")
        end
        h5open(path, "r") do file
            physical = file["spectrum/physical"]
            for name in ("m", "mode_indices", "fK_log10", "fpert_log10",
                    "lambda_self_sign", "lambda_self_log10", "lambda_31_sign",
                    "lambda_31_log10", "lambda_31_indices", "lambda_22_sign",
                    "lambda_22_log10", "lambda_22_indices")
                haskey(physical, name) && (out[key * "/F1/" * name] = read(physical[name]))
            end
        end
    end
    out
end

function r005_axion_photon_routes!(out, fixture, kinv_scale)
    key = fixture.name
    ap = CYAxiverse.axion_photon
    policies = fixture.visible === nothing ? (:electron_proxy,) :
        (:electron_proxy, :divisor_instanton)
    r005_with_geometry(fixture, kinv_scale) do geom_idx, path, _
        for policy in policies
            result = ap._run_local_scan(path; qed_threshold_policy=policy)
            prefix = key * "/F3_" * string(policy)
            h = result.hierarchy
            p = result.photons
            out[prefix * "/q"] = h.q
            out[prefix * "/theta_from_canonical"] = h.theta_from_canonical
            out[prefix * "/log10_f_GeV"] = h.log10_f_GeV
            out[prefix * "/log10_mass_eV"] = h.log10_mass_eV
            out[prefix * "/mixing_theta"] = p.theta
            out[prefix * "/log10_g_GeVinv"] = p.log10_g_GeVinv
            out[prefix * "/log10_g_effective_GeVinv"] = p.log10_g_effective_GeVinv
            out[prefix * "/log10_photon_width_GeV"] = p.log10_photon_width_GeV
            out[prefix * "/log10_quartic_width_GeV"] = p.log10_quartic_width_GeV
            out[prefix * "/log10_light_threshold_eV"] = p.log10_light_threshold_eV
            out[prefix * "/light_mode_count"] = p.light_mode_count
            # Persisted hierarchy datasets (axion_photon.jl:1382–1385).
            ap.write_axion_photon_result(path, result; force=true)
            restored = ap.read_axion_photon_result(path)
            out[prefix * "/persisted/q"] = restored.hierarchy.q
            out[prefix * "/persisted/theta_from_canonical"] =
                restored.hierarchy.theta_from_canonical
            out[prefix * "/persisted/log10_f_GeV"] = restored.hierarchy.log10_f_GeV
            out[prefix * "/persisted/log10_mass_eV"] = restored.hierarchy.log10_mass_eV
        end
    end
    out
end

function r005_inflation_routes!(out, fixture, kinv_scale)
    key = fixture.name
    points = CYAxiverse.inflation_points
    r005_with_geometry(fixture, kinv_scale) do geom_idx, path, _
        # In memory: diagnose at non-stationary points.
        prepared = points.prepare_geometry_context(geom_idx)
        for (k, theta) in enumerate(fixture.theta_points)
            d = points.diagnose(prepared.context, theta)
            prefix = key * "/inflation/point$k"
            out[prefix * "/epsilon"] = d.epsilon
            out[prefix * "/eta"] = d.eta_values
            out[prefix * "/eigenvalues"] = d.hessian_eigenvalues
            out[prefix * "/gradient_norm"] = d.gradient_norm
            out[prefix * "/value"] = d.value
            out[prefix * "/mode_counts"] =
                [d.negative_modes, d.zeroish_modes, d.positive_modes]
        end
        # Diagnostic CSV row fields (inflation_scan_common.jl run_geometry).
        row = run_geometry(geom_idx; max_branches=1_000)
        prefix = key * "/inflation/run_geometry"
        out[prefix * "/mass_min"] = row.mass_min
        out[prefix * "/mass_max"] = row.mass_max
        out[prefix * "/candidate_slowroll_saddles"] = row.candidate_slowroll_saddles
        out[prefix * "/saddle_count"] = row.saddle_count
        out[prefix * "/leading_minima_count"] = row.leading_minima_count
        out[prefix * "/negative_mass_count"] = row.negative_mass_count
        # F6 through write_inflation_group!, e-fold scan not run.
        write_inflation_group!(geom_idx, (; status=:success, result=row),
            (; status=:not_run, message="R-005 harness: e-fold scan not run");
            catastrophe_settings=(; max_branches=1_000),
            efold_settings=(; precision_bits=128), force=true)
        h5open(path, "r") do file
            catastrophes = file["inflation/catastrophes"]
            for name in ("mass_min", "mass_max", "candidate_slowroll_saddles",
                    "leading_minima_count", "leading_branch_saddle_count")
                out[key * "/F6/" * name] = read(catastrophes[name])
            end
        end
    end
    out
end

function r005_model_stage_routes!(out, fixture, kinv_scale)
    key = fixture.name
    PB = CYAxiverse.paper_benchmarks
    reference = PB.leading_axion_reference_data(fixture.Q, fixture.tau,
        fixture.cy_volume, fixture.prefactor, fixture.gravitino,
        kinv_scale .* fixture.Kinv)
    out[key * "/F4/mass_log10_ev_reference"] = reference.mass_log10_ev_reference
    out[key * "/F4/divisor_index"] = reference.divisor_index
    out
end

"""
    r005_route_outputs(; kinv_scale=1)

Evaluate every R-005 route on both fixtures with the stored inverse metric
`kinv_scale · Kinv`. `kinv_scale = 2` is the c = 1/2 evaluation.
"""
function r005_route_outputs(; kinv_scale::Real=1)
    out = Dict{String, Any}()
    for fixture in r005_fixtures()
        for (route, run!) in ((:spectrum, r005_spectrum_routes!),
                (:axion_photon, r005_axion_photon_routes!),
                (:inflation, r005_inflation_routes!),
                (:model_stage, r005_model_stage_routes!))
            route in fixture.routes || continue
            r005_route!(() -> run!(out, fixture, kinv_scale), out,
                fixture.name * "/" * string(route))
        end
    end
    out
end
