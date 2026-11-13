# CYAX-0232: the c = 1 supergravity axion kinetic normalization.
# R-003 (named source reproduction), R-004 (factor contract and orientation)
# and R-005 (scaling contract against a c = 1/2 evaluation of the same inputs).
# Fixture inputs are pinned in specs/0232-kinetic-normalization/plan.md.

using LinearAlgebra

# The R-005 comparison harness (T013): `r005_route_outputs(; kinv_scale)` and
# the Eq. 3.18 helper `r005_eq318`. It needs `run_batch`, `run_geometry` and
# `write_inflation_group!`, which runtests.jl includes from scripts/.
include(joinpath(@__DIR__, "..", "scripts", "r005_scaling_routes.jl"))

_cyax0232_eq318_log10_lambda4(tau, cy_volume, gs, w0) =
    r005_eq318(tau, cy_volume; gs, w0)

@testset "CYAX-0232 R-004 kinetic factor contract and orientation" begin
    read_api = CYAxiverse.read
    @test CYAxiverse.KINETIC_NORMALIZATION === :sugra_c1
    @test kinetic_factor === read_api.kinetic_factor
    @test kinetic_matrix === read_api.kinetic_matrix

    # Factor identities on the non-diagonal fixture.
    Kinv = [3.0 1.0; 1.0 2.0]
    B = kinetic_factor(Kinv)
    K = kinetic_matrix(Kinv)
    @test norm(B * B' - Kinv / 2) <= 1e-14 * norm(Kinv)
    @test norm(B' * K * B - I) <= 1e-12
    @test istril(Matrix(B))
    @test all(>(0), diag(B))
    @test K ≈ 2 * inv(Kinv)
    # The factor ½ is applied only inside the shared functions.
    @test B ≈ cholesky(Symmetric(Kinv)).L / sqrt(2)

    # Non-diagonal fixture: charges q₁ = (1, 0) and q₂ = (1, 1) lead, q₃ = (0, 1)
    # is subleading; τ₁ < τ₂ so q₁ leads, and Λ₂⁴/Λ₁⁴ ≤ 1e-20.
    Q = Int[1 1 0; 0 1 1]
    tau = [1.0, 20.0, 30.0]
    cy_volume = 50.0
    gs, w0 = 0.5029733, 1.0
    eq318 = _cyax0232_eq318_log10_lambda4(tau, cy_volume, gs, w0)
    L = eq318.L
    @test L[2, 2] - L[2, 1] <= -20

    # Closed form: G = Qᵀ (Kinv/2) Q on the leading pair, Gram–Schmidt norms
    # ĝ₁ = sqrt(G₁₁), ĝ₂ = sqrt(det G / G₁₁).
    Qlead = Q[:, 1:2]
    G = Qlead' * (Kinv / 2) * Qlead
    @test G ≈ [1.5 2.0; 2.0 3.5]
    ghat = [sqrt(G[1, 1]), sqrt(det(G) / G[1, 1])]
    @test ghat ≈ [1.224744871, 0.912870929] atol=1e-9

    # Through axion_photon.leading_hierarchy (f and m).
    ap = CYAxiverse.axion_photon
    hierarchy = ap.leading_hierarchy(
        ap.InstantonData(Q, L[2, :], Int.(L[1, :]), [1, 2, 3]), Kinv)
    @test hierarchy.selected_indices == [1, 2]
    log10_mp = log10(ap.M_PLANCK_GEV)
    expected_log10_f = log10_mp .- log10(2π) .- log10.(ghat)
    expected_log10_m = 0.5 .* L[2, 1:2] .+ log10_mp .+ 9 .+ log10(2π) .+ log10.(ghat)
    @test isapprox(hierarchy.log10_f_GeV, expected_log10_f; rtol=1e-10)
    @test isapprox(hierarchy.log10_mass_eV, expected_log10_m; rtol=1e-10)
    # m² = Λ⁴/f², in eV and GeV: log10 m = ½ log10 Λ⁴ + 2 log10 M_pl - log10 f + 9.
    @test isapprox(hierarchy.log10_mass_eV,
        0.5 .* L[2, 1:2] .+ 2log10_mp .- hierarchy.log10_f_GeV .+ 9; rtol=1e-10)

    # Through the Julia model stage (m only).
    PB = CYAxiverse.paper_benchmarks
    reference = PB.leading_axion_reference_data(
        Q, tau, cy_volume, eq318.prefactor, eq318.gravitino, Kinv)
    @test reference.divisor_index == [1, 2]
    log10_mp_stage = Float64(log10(CYAxiverse.generate.constants()["MPlanck"]))
    stage_expected = 0.5 .* L[2, 1:2] .+ log10_mp_stage .+ 9 .+ log10(2π) .+ log10.(ghat)
    @test isapprox(reference.mass_log10_ev_reference, stage_expected; rtol=1e-10)
    @test reference.mass_log10_ev_reference ≈ [24.258982913, -1.141289065] atol=1e-8

    # R-005: the same route with 2 Kinv in place of Kinv is c = 1/2, and every
    # mass moves by exactly ½ log10 2 (m × 1/√2 at c = 1).
    c_half = PB.leading_axion_reference_data(
        Q, tau, cy_volume, eq318.prefactor, eq318.gravitino, 2 * Kinv)
    @test c_half.mass_log10_ev_reference ≈ [24.409497911, -0.990774067] atol=1e-8
    @test isapprox(c_half.mass_log10_ev_reference .- reference.mass_log10_ev_reference,
        fill(0.5 * log10(2.0), 2); atol=1e-10)

    # Orientation guard: B Q in place of Bᵀ Q changes an asserted value.
    wrong = abs.(diag(qr(Matrix(B) * Qlead).R))
    @test wrong ≈ [1.29099, 0.86603] atol=1e-5
    @test maximum(abs.(wrong ./ ghat .- 1)) > 1e-3

    # Scalar fixture: f = M_pl sqrt(2 / (q² k)) / (2π) with k the scalar Kinv,
    # through both shared functions and through leading_hierarchy.
    k, q = 0.25, 3
    kinv1 = reshape([k], 1, 1)
    @test kinetic_factor(kinv1)[1, 1]^2 ≈ k / 2
    @test kinetic_matrix(kinv1)[1, 1] ≈ 2 / k
    f_scalar = ap.M_PLANCK_GEV * sqrt(2 / (q^2 * k)) / (2π)
    @test f_scalar ≈ ap.M_PLANCK_GEV / (2π * q * kinetic_factor(kinv1)[1, 1])
    @test f_scalar ≈ ap.M_PLANCK_GEV * sqrt(kinetic_matrix(kinv1)[1, 1]) / (2π * q)
    one_axion = ap.leading_hierarchy(
        ap.InstantonData(reshape([q], 1, 1), [-20.0], [1], [1]), kinv1)
    @test one_axion.log10_f_GeV[1] ≈ log10(f_scalar) rtol=1e-12
end

@testset "CYAX-0232 R-003 Fuzzy Axions h11=2 source reproduction" begin
    # Literal inputs of "Fuzzy-axion h11=2 source-corrected role validation";
    # arXiv:2412.12012 Sec. 4.2.1, Eq. 4.3.
    Q = Int[7 1 1 2 3 0; 2 0 0 0 1 1]
    tau = Float64[312.28056480059246, 34.041122071663615, 34.041122071663615,
        68.08224414332723, 139.11972136446443, 36.99635514947357]
    cy_volume = 222.48322433885366
    inverse_metric = Float64[2238.718569786352 -2617.525402950724;
        -2617.525402950724 28440.2666884343]
    eq318 = _cyax0232_eq318_log10_lambda4(tau, cy_volume, 0.5029733, 1.0)
    mplanck = Float64(CYAxiverse.generate.constants()["MPlanck"])

    function source_observables(kinv)
        # QCD first (D6), then fuzzy (D2).
        Q_reduced = Q[:, [6, 2]]
        q, _, _ = CYAxiverse.axion_photon._canonical_frame(Q_reduced, kinv)
        f_qcd, f_fuzzy = mplanck ./ (2π .* diag(q))
        lambda2_fuzzy = mplanck^2 * 10.0^(0.5 * eq318.L[2, 2])
        m_fuzzy = 1e9 * lambda2_fuzzy / f_fuzzy
        m_qcd = 5.7e-10 * (1e16 / f_qcd)  # Eq. 2.6
        (; f_qcd, f_fuzzy, m_qcd, m_fuzzy)
    end
    within(x, target, tol) = abs(x / target - 1) <= tol

    c1 = source_observables(inverse_metric)
    @test within(c1.f_qcd, 3.2e15, 0.05)
    @test within(c1.f_fuzzy, 1.2e16, 0.05)
    @test within(c1.m_qcd, 1.8e-9, 0.05)
    @test within(c1.m_fuzzy, 5.0e-20, 0.05)

    # Discrimination: c = 1/2 (Kinv in place of Kinv/2, i.e. 2 Kinv through
    # kinetic_factor) misses each decay constant by more than 20%.
    c_half = source_observables(2 * inverse_metric)
    @test !within(c_half.f_qcd, 3.2e15, 0.20)
    @test !within(c_half.f_fuzzy, 1.2e16, 0.20)
    @test !within(c_half.m_qcd, 1.8e-9, 0.05)
end

# R-005 rows (T014–T016). One c = 1 and one c = 1/2 evaluation of every route;
# `h = ½ log10 2`. Log10 fields are compared by offset (1e-10 absolute),
# linear fields by ratio (1e-10 relative, normwise). Counts and
# classifications marked "reported" in the spec are reported by
# `scripts/r005_scaling_harness.jl report`, not asserted here.
const R005_H = 0.5 * log10(2.0)
const R005_C1 = r005_route_outputs(; kinv_scale=1)
const R005_CHALF = r005_route_outputs(; kinv_scale=2)

r005_array(x) = x isa Number ? [Float64(x)] : Float64.(x)

function r005_offset(key, offset; c1=R005_C1, chalf=R005_CHALF)
    a, b = r005_array(c1[key]), r005_array(chalf[key])
    size(a) == size(b) || return false
    finite = isfinite.(a)
    finite == isfinite.(b) || return false
    a[.!finite] == b[.!finite] || return false
    any(finite) || return false
    maximum(abs.(a[finite] .- b[finite] .- offset)) <= 1e-10
end

function r005_ratio(key, ratio; c1=R005_C1, chalf=R005_CHALF)
    a, b = r005_array(c1[key]), r005_array(chalf[key])
    size(a) == size(b) || return false
    norm(a .- ratio .* b) <= 1e-10 * norm(ratio .* b)
end

r005_same(key) = R005_C1[key] == R005_CHALF[key]

@testset "CYAX-0232 R-005 harness" begin
    # No route fails. (Before the A1 orientation fix, T030, hp_spectrum_save
    # failed on the R-003 charges, where :hp_effective keeps N > h11.)
    @test isempty([k for k in keys(R005_C1) if endswith(k, "/error")])
    @test isempty([k for k in keys(R005_CHALF) if endswith(k, "/error")])
    @test Set(keys(R005_C1)) == Set(keys(R005_CHALF))
    # Both F2 writers label their output (R-006).
    for fixture in ("r004", "r003", "infl"), route in ("F2_hp", "F2_pq")
        @test R005_C1["$fixture/$route/metadata/schema_version"] ==
            "cyaxiverse-spectrum-groups-1"
        @test R005_C1["$fixture/$route/metadata/kinetic_normalization"] == "sugra_c1"
        @test r005_same("$fixture/$route/metadata/kinetic_normalization")
    end
end

@testset "CYAX-0232 R-005 spectrum, F1 and F2 (T014)" begin
    for fixture in ("r004", "r003", "infl")
        # F1: physical-spectrum batch runner. The threshold-filtered mode set
        # is the same in both conventions for these fixtures.
        @test r005_same("$fixture/F1/mode_indices")
        @test r005_offset("$fixture/F1/m", -R005_H)
        @test r005_offset("$fixture/F1/fK_log10", R005_H)
        @test r005_offset("$fixture/F1/fpert_log10", R005_H)
        for name in ("lambda_self", "lambda_31", "lambda_22")
            @test r005_offset("$fixture/F1/$(name)_log10", -4R005_H)
            @test r005_same("$fixture/F1/$(name)_sign")
        end

        # F2 in memory: hp_spectrum and pq_spectrum.
        for route in ("F2_hp_memory", "F2_pq_memory")
            @test r005_offset("$fixture/$route/m", -R005_H)
            @test r005_offset("$fixture/$route/fK", R005_H)
            for name in ("λself", "λ31", "λ22")
                @test r005_offset("$fixture/$route/$name", -4R005_H)
                @test r005_same("$fixture/$route/$(name)sign")
            end
        end
        @test r005_offset("$fixture/F2_hp_memory/fpert", R005_H)
        @test r005_offset("$fixture/F2_pq_memory/f", R005_H)

        # F2 persisted: pq_spectrum_save (all fixtures) and hp_spectrum_save.
        @test r005_offset("$fixture/F2_pq/masses/log10", -R005_H)
        @test r005_offset("$fixture/F2_pq/decay/fK", R005_H)
        @test r005_offset("$fixture/F2_pq/decay/fpert", R005_H)
        @test r005_offset("$fixture/F2_hp/masses/log10", -R005_H)
        @test r005_offset("$fixture/F2_hp/decay/fK", R005_H)
        @test r005_offset("$fixture/F2_hp/decay/fpert", R005_H)
        for group in ("quartdiag", "quart31", "quart22")
            @test r005_offset("$fixture/F2_hp/$group/log10", -4R005_H)
            @test r005_same("$fixture/F2_hp/$group/sign")
        end
        # The hp_spectrum_save cubic is in the ϑ/charge basis: factor 1. The
        # R-005 cubic rows are asserted on the A1.4 cubic fixtures (T030).
        @test r005_same("$fixture/F2_hp/cubic/tensor")
    end
end

@testset "CYAX-0232 R-005 fuzzy-axion model stage, F4 (T014)" begin
    for fixture in ("r004", "r003")
        @test r005_same("$fixture/F4/divisor_index")
        @test r005_offset("$fixture/F4/mass_log10_ev_reference", -R005_H)
    end
end

@testset "CYAX-0232 R-005 axion–photon, F3 (T015)" begin
    for (fixture, policy) in (("r004", "electron_proxy"),
            ("r004", "divisor_instanton"), ("r003", "electron_proxy"))
        prefix = "$fixture/F3_$policy"
        @test r005_offset("$prefix/log10_f_GeV", R005_H)
        @test r005_offset("$prefix/log10_mass_eV", -R005_H)
        @test r005_ratio("$prefix/mixing_theta", 1.0)
        @test r005_ratio("$prefix/q", 1 / sqrt(2))
        @test r005_ratio("$prefix/theta_from_canonical", 1 / sqrt(2))
        @test r005_offset("$prefix/log10_g_GeVinv", -R005_H)
        @test r005_offset("$prefix/log10_quartic_width_GeV", -9R005_H)
        # Persisted hierarchy datasets (axion_photon.jl:1382–1385).
        @test r005_ratio("$prefix/persisted/q", 1 / sqrt(2))
        @test r005_ratio("$prefix/persisted/theta_from_canonical", 1 / sqrt(2))
        @test r005_offset("$prefix/persisted/log10_f_GeV", R005_H)
        @test r005_offset("$prefix/persisted/log10_mass_eV", -R005_H)
    end

    # :divisor_instanton: the threshold moves with the QED kinetic norm, so
    # every mode keeps its side and g_eff, widths and light-mode count follow
    # the all-mode factors.
    prefix = "r004/F3_divisor_instanton"
    @test r005_offset("$prefix/log10_light_threshold_eV", -R005_H)
    @test r005_offset("$prefix/log10_g_effective_GeVinv", -R005_H)
    @test r005_offset("$prefix/log10_photon_width_GeV", -5R005_H)
    @test r005_same("$prefix/light_mode_count")
    @test R005_C1["$prefix/light_mode_count"] == 1

    # :electron_proxy: the threshold is the electron mass in both
    # conventions; light and heavy modes take different factors. Modes that
    # cross the threshold are reported (none for these fixtures).
    for fixture in ("r004", "r003")
        prefix = "$fixture/F3_electron_proxy"
        @test r005_offset("$prefix/log10_light_threshold_eV", 0.0)
        threshold = R005_C1["$prefix/log10_light_threshold_eV"]
        light_c1 = R005_C1["$prefix/log10_mass_eV"] .<= threshold
        light_chalf = R005_CHALF["$prefix/log10_mass_eV"] .<= threshold
        @test light_c1 == light_chalf
        @test r005_same("$prefix/light_mode_count")
        expected_g = ifelse.(light_c1, -3R005_H, -R005_H)
        expected_width = ifelse.(light_c1, -9R005_H, -5R005_H)
        g1, gh = R005_C1["$prefix/log10_g_effective_GeVinv"],
            R005_CHALF["$prefix/log10_g_effective_GeVinv"]
        w1, wh = R005_C1["$prefix/log10_photon_width_GeV"],
            R005_CHALF["$prefix/log10_photon_width_GeV"]
        @test maximum(abs.(g1 .- gh .- expected_g)) <= 1e-10
        @test maximum(abs.(w1 .- wh .- expected_width)) <= 1e-10
    end
    # The R-004 F3 fixture has one mode on each side of the electron mass.
    @test R005_C1["r004/F3_electron_proxy/light_mode_count"] == 1
    @test R005_C1["r003/F3_electron_proxy/light_mode_count"] == 2
end

@testset "CYAX-0232 R-005 inflation and F6 (T016)" begin
    for point in 1:3
        prefix = "infl/inflation/point$point"
        # Non-stationary evaluation points.
        @test R005_C1["$prefix/gradient_norm"] > 1e-3
        @test r005_ratio("$prefix/value", 1.0)
        @test r005_ratio("$prefix/epsilon", 0.5)
        @test r005_ratio("$prefix/eta", 0.5)
        @test r005_ratio("$prefix/eigenvalues", 0.5)
    end
    # Diagnostic CSV row fields from run_geometry and the F6 catastrophes
    # group written by write_inflation_group!.
    for key in ("infl/inflation/run_geometry/mass_min",
            "infl/inflation/run_geometry/mass_max",
            "infl/F6/mass_min", "infl/F6/mass_max")
        @test r005_offset(key, -R005_H)
    end
end

@testset "CYAX-0232 R-001 K-argument docstrings (T020)" begin
    _doc(m, s) = string(Base.Docs.doc(Base.Docs.Binding(m, s)))
    gen = CYAxiverse.generate
    # Every API that takes K (approval note 1 included) states the contract.
    k_apis = [(gen, s) for s in (:hp_spectrum, :leading_hessian_matrix_float64,
        :leading_hessian_matrix_float64_scaled, :high_precision_leading_hessian,
        :mass_basis_accuracy, :instanton_hierarchy_diagnostics,
        :leading_hessian_mass_basis, :leading_hessian_mass_basis_float64,
        :pq_spectrum, :pq_physical_spectrum, :pq_hybrid_physical_spectrum,
        :pq_window_spectrum, :pq_physical_mode_count,
        :physical_mode_inertia_count, :confirm_physical_mode_count,
        :pq_schur_admissible, :pq_hp_alignment)]
    append!(k_apis, [(CYAxiverse.inflation_points, :prepare_context),
        (CYAxiverse.inflation_points, :compare_precision),
        (CYAxiverse.paper_benchmarks,
            :local_catastrophe_diagnostic)])
    for (m, s) in k_apis
        @test occursin("K-argument", _doc(m, s))
    end
    # APIs that take a canonical factor C state its contract.
    for s in (:leading_hessian_matrix_float64, :leading_hessian_matrix_float64_scaled,
            :high_precision_leading_hessian, :mass_basis_accuracy,
            :instanton_hierarchy_diagnostics, :leading_hessian_mass_basis,
            :leading_hessian_mass_basis_float64, :physical_mode_inertia_count,
            :confirm_physical_mode_count)
        @test occursin("Canonical-factor contract", _doc(gen, s))
    end
    # read.K is kept, documented as unlabelled and not the physical accessor.
    k_doc = _doc(CYAxiverse.read, :K)
    @test occursin("not the physical kinetic-matrix accessor", k_doc)
    @test occursin("kinetic_matrix", k_doc)
    for s in (:potential, :oriented_potential)
        @test occursin("2 Kinv⁻¹", _doc(CYAxiverse.read, s))
    end
end

# A1.4 cubic fixtures (plan.md, Cubic fixtures). L columns are (sign, log10 Λ).
const A1_FIXTURES = (
    (name="C2", h11=2, np=21, Q=Int[1 1 0; 0 1 1], L=Float64[1 1 1; 0 -0.5 -1.0],
     Kinv=Float64[3 1; 1 2], theta0=[0.1, 0.05], tau=ones(3), cy_volume=10.0,
     visible=nothing, writers=(:hp, :pq), hp_count=2),
    (name="C2-hp", h11=2, np=22, Q=Int[1 1 0; 0 1 1], L=Float64[1 1 1; 0 -0.5 -0.7],
     Kinv=Float64[3 1; 1 2], theta0=[0.1, 0.05], tau=ones(3), cy_volume=10.0,
     visible=nothing, writers=(:hp,), hp_count=3),
    (name="C3", h11=3, np=23, Q=Int[1 0 0 1; 0 1 0 1; 0 0 1 1],
     L=Float64[1 1 1 1; 0 -0.4 -0.9 -1.5],
     Kinv=Float64[3 1 0; 1 2 0.5; 0 0.5 1.5], theta0=[0.1, 0.05, 0.02],
     tau=ones(4), cy_volume=10.0, visible=nothing, writers=(:pq,), hp_count=0),
)

# Hessian of V(x) = Σₐ Λₐ (1 - cos(Qrows[a, :]·x)), independent of generate.
a1_hessian(x, Λ, Qrows) = sum(Λ[a] * cos(dot(Qrows[a, :], x)) *
    (Qrows[a, :] * Qrows[a, :]') for a in eachindex(Λ))

function a1_fd_cubic(x, Λ, Qrows; h=1e-5)
    n = length(x)
    T = zeros(n, n, n)
    for k in 1:n
        e = zeros(n)
        e[k] = h
        T[:, :, k] = (a1_hessian(x .+ e, Λ, Qrows) .- a1_hessian(x .- e, Λ, Qrows)) ./ (2h)
    end
    T
end

a1_close(stored, reference) =
    maximum(abs.(stored .- reference)) <= 1e-6 * maximum(abs.(stored))

function a1_save(fixture, writer, kinv_scale)
    r005_with_geometry(fixture, kinv_scale) do geom_idx, path, _
        save = writer === :hp ? CYAxiverse.generate.hp_spectrum_save :
            CYAxiverse.generate.pq_spectrum_save
        save(fixture.h11, fixture.np, 1; phase=fixture.theta0)
        h5open(path, "r") do file
            metadata = file["spectrum/metadata"]
            (; tensor=read(file["spectrum/cubic/tensor"]),
                phase=read(file["spectrum/cubic/phase"]),
                metadata=Dict(k => read(metadata[k]) for k in keys(metadata)),
                cubic=CYAxiverse.read.cubic_tensor(geom_idx),
                masses=CYAxiverse.read.pq_spectrum(fixture.h11, fixture.np, 1),
                hp=writer === :hp ?
                    CYAxiverse.read.hp_spectrum(fixture.h11, fixture.np, 1) : nothing)
        end
    end
end

# Mass-basis data of pq_spectrum_save, rebuilt from the stored inverse metric.
function a1_mass_frame(fixture, kinv_scale)
    C = Matrix(kinetic_factor(kinv_scale .* fixture.Kinv))
    selected = CYAxiverse.generate.LQtilde(fixture.Q, fixture.L)
    _, _, basis = CYAxiverse.generate.leading_hessian_mass_basis_float64(C,
        selected.Ltilde, selected.Qtilde)
    (; C, basis, Ltilde=selected.Ltilde, Qtilde=selected.Qtilde,
        Qmass=(Matrix(selected.Qtilde') * C) * basis)
end

@testset "CYAX-0232 A1 F2 cubic tensor (T030)" begin
    for fixture in A1_FIXTURES
        # A1.4: every |Λₐ| within 10^1.5 of the others.
        @test maximum(fixture.L[2, :]) - minimum(fixture.L[2, :]) <= 1.5
        @test all(==(1), fixture.L[1, :])
        if :hp in fixture.writers
            out = a1_save(fixture, :hp, 1)
            Lsel, Qsel = CYAxiverse.generate._hp_selected_potential(fixture.L,
                fixture.Q, :hp_effective)
            @test size(Qsel, 2) == fixture.hp_count
            if fixture.name == "C2-hp"
                canonical = CYAxiverse.generate.αmatrix(
                    CYAxiverse.generate.LQtilde(fixture.Q, fixture.L))
                @test canonical isa CYAxiverse.structs.Canonicalα
                @test size(Qsel, 2) > fixture.h11
            end
            @test out.phase == fixture.theta0
            @test out.metadata == Dict("schema_version" => "cyaxiverse-spectrum-groups-1",
                "kinetic_normalization" => "sugra_c1", "cubic_basis" => "charge_basis")
            @test out.cubic.cubic_basis === :charge_basis
            @test out.cubic.kinetic_normalization === :sugra_c1
            @test out.cubic.kinetic_normalization_inferred === false
            @test out.hp.kinetic_normalization === :sugra_c1
            @test out.masses.kinetic_normalization_inferred === false
            Λ = Lsel[1, :] .* 10 .^ Lsel[2, :]
            @test maximum(abs.(out.tensor)) > 0
            @test a1_close(out.tensor, a1_fd_cubic(fixture.theta0, Λ, Matrix(Qsel')))

            # R-005: the charge-basis cubic does not depend on K.
            half = a1_save(fixture, :hp, 2)
            @test norm(out.tensor .- half.tensor) <= 1e-10 * norm(half.tensor)
        end
        if :pq in fixture.writers
            out = a1_save(fixture, :pq, 1)
            frame = a1_mass_frame(fixture, 1)
            # Non-degenerate leading masses.
            W = CYAxiverse.generate.leading_hessian_matrix_float64(frame.C,
                frame.Ltilde, frame.Qtilde)
            λ = sort(eigvals(W))
            @test all(>(0), λ)
            @test all((λ[k + 1] - λ[k]) / λ[k + 1] >= 0.5 for k in 1:length(λ) - 1)

            @test out.phase == fixture.theta0
            @test out.metadata["cubic_basis"] == "canonical_mass_basis"
            @test out.metadata["kinetic_normalization"] == "sugra_c1"
            @test out.cubic.cubic_basis === :canonical_mass_basis
            # x₀ from the stored ϑ₀, and Qmass·x₀ = Q̃ᵀ·ϑ₀ (A1.2).
            x0 = frame.basis' * (frame.C \ out.phase)
            @test frame.Qmass * x0 ≈ frame.Qtilde' * fixture.theta0 rtol=1e-12
            Λ = frame.Ltilde[1, :] .* 10 .^ frame.Ltilde[2, :]
            @test maximum(abs.(out.tensor)) > 0
            @test a1_close(out.tensor, a1_fd_cubic(x0, Λ, frame.Qmass))

            # R-005: mass-basis cubic ×2^{-3/2} after per-mode sign alignment.
            half = a1_save(fixture, :pq, 2)
            frame_half = a1_mass_frame(fixture, 2)
            signs = sign.(diag(frame.basis' * frame_half.basis))
            @test all(!iszero, signs)
            aligned = [signs[i] * signs[j] * signs[k] * half.tensor[i, j, k]
                for i in axes(half.tensor, 1), j in axes(half.tensor, 2),
                    k in axes(half.tensor, 3)]
            expected = 2.0^(-1.5) .* aligned
            @test norm(out.tensor .- expected) <= 1e-10 * norm(expected)
        end
    end
end

@testset "CYAX-0232 F2 labels, replace and refusal (T030)" begin
    fixture = A1_FIXTURES[1]
    f2_label = CYAxiverse.read.f2_label
    function legacy_f2!(path)
        h5open(path, "r+") do file
            spectrum = create_group(file, "spectrum")
            spectrum["masses/log10"] = [1.0, 2.0]
            spectrum["masses/sign"] = [1, 1]
            spectrum["decay/fK"] = [3.0, 4.0]
            spectrum["decay/fpert"] = [3.0, 4.0]
            spectrum["cubic/tensor"] = zeros(2, 2, 2)
            spectrum["cubic/phase"] = [0.3, 0.0]
            spectrum["physical/m"] = [5.0]
            spectrum["axion_photon/identity/schema_version"] = 1
        end
    end
    r005_with_geometry(fixture, 1) do geom_idx, path, _
        legacy_f2!(path)
        # Legacy F2: inferred :legacy_c_half, cubic basis unknown.
        @test CYAxiverse.read.stored_kinetic_normalization(path; family=:f2) ==
            (:legacy_c_half, true)
        cubic = CYAxiverse.read.cubic_tensor(geom_idx)
        @test cubic.cubic_basis === :unknown
        @test cubic.kinetic_normalization === :legacy_c_half
        @test cubic.kinetic_normalization_inferred === true
        @test_throws ArgumentError CYAxiverse.read.require_cubic_basis(cubic)
        legacy = CYAxiverse.read.pq_spectrum(fixture.h11, fixture.np, 1)
        @test (legacy.kinetic_normalization, legacy.kinetic_normalization_inferred) ==
            (:legacy_c_half, true)

        # Both writers refuse legacy F2 groups without replace.
        @test_throws ArgumentError CYAxiverse.generate.pq_spectrum_save(fixture.h11,
            fixture.np, 1; phase=fixture.theta0)
        @test_throws ArgumentError CYAxiverse.generate.hp_spectrum_save(fixture.h11,
            fixture.np, 1; phase=fixture.theta0)
        @test h5open(file -> read(file["spectrum/masses/log10"]), path) == [1.0, 2.0]

        # replace=true removes only the F2 groups; F1 and F3 are intact.
        CYAxiverse.generate.pq_spectrum_save(fixture.h11, fixture.np, 1;
            phase=fixture.theta0, replace=true)
        h5open(path, "r") do file
            @test read(file["spectrum/physical/m"]) == [5.0]
            @test read(file["spectrum/axion_photon/identity/schema_version"]) == 1
            @test f2_label(file["spectrum"], path) == (:sugra_c1, false,
                :canonical_mass_basis)
        end
        # A current "sugra_c1" output is replaced without the option, and no
        # dataset of the earlier writer survives.
        CYAxiverse.generate.hp_spectrum_save(fixture.h11, fixture.np, 1;
            phase=fixture.theta0)
        CYAxiverse.generate.pq_spectrum_save(fixture.h11, fixture.np, 1;
            phase=fixture.theta0)
        h5open(path, "r") do file
            spectrum = file["spectrum"]
            @test !haskey(spectrum, "quartdiag")
            @test sort(collect(keys(spectrum))) ==
                ["axion_photon", "cubic", "decay", "masses", "metadata", "physical"]
        end
    end

    # R-007 rejections for a current-version F2.
    function rejected(edit!)
        r005_with_geometry(fixture, 1) do geom_idx, path, _
            CYAxiverse.generate.pq_spectrum_save(fixture.h11, fixture.np, 1;
                phase=fixture.theta0)
            h5open(edit!, path, "r+")
            reader = try
                CYAxiverse.read.cubic_tensor(geom_idx)
                false
            catch err
                err isa ArgumentError && occursin(path, sprint(showerror, err))
            end
            writer = try
                CYAxiverse.generate.pq_spectrum_save(fixture.h11, fixture.np, 1;
                    phase=fixture.theta0)
                false
            catch err
                err isa ArgumentError && occursin("replace=true", sprint(showerror, err))
            end
            reader && writer
        end
    end
    set!(name, value) = file -> begin
        metadata = file["spectrum/metadata"]
        HDF5.delete_object(metadata, name)
        value === nothing || (metadata[name] = value)
    end
    @test rejected(set!("kinetic_normalization", nothing))
    @test rejected(set!("kinetic_normalization", "bogus"))
    @test rejected(set!("kinetic_normalization", "legacy_c_half"))
    @test rejected(set!("kinetic_normalization", "p96_reproduction_c_half"))
    @test rejected(set!("cubic_basis", nothing))
    @test rejected(set!("cubic_basis", "mass_basis"))
    @test rejected(set!("schema_version", "cyaxiverse-spectrum-groups-2"))
    @test_throws ArgumentError CYAxiverse.read.stored_kinetic_normalization("x.h5";
        family=:f9)
end

@testset "CYAX-0232 F1 label, rewrite and summary CSV (T031)" begin
    r = CYAxiverse.read
    spectrum = CYAxiverse.structs.PhysicalAxionSpectrum(
        [1.0, 2.0], [0, 1], zeros(2, 2), [1, -1], [3.0, 4.0],
        reshape([1, 1, 1, 2], 4, 1), [1], [0.5],
        reshape([1, 1, 2, 2], 4, 1), [-1], [0.7],
        -30.0, 200)
    geom_idx = CYAxiverse.structs.GeometryIndex(2, 3, 4)
    write_f1(path; quartics=true) = _write_result(path, geom_idx, spectrum;
        prec=200, threshold_log10=-30.0, quartics, runtime_seconds=0.1,
        provisional=false, fK=[5.0, 6.0])
    function f1_fixture(f; quartics=true)
        mktempdir() do root
            path = joinpath(root, "h11_002", "np_0000003", "cy_0000004", "cyax.h5")
            mkpath(dirname(path))
            h5open(file -> create_group(file, "cytools"), path, "cw")
            write_f1(path; quartics)
            withenv("CYAXIVERSE_DATA_DIR" => root) do
                f(path)
            end
        end
    end
    metadata(file) = file["spectrum/physical/metadata"]
    set!(name, value) = file -> begin
        m = metadata(file)
        haskey(m, name) && HDF5.delete_object(m, name)
        value === nothing || (m[name] = value)
    end
    labels(x) = (x.kinetic_normalization, x.kinetic_normalization_inferred)

    # Round trip of a current output.
    f1_fixture() do path
        h5open(path, "r") do file
            @test read(metadata(file)["schema_version"]) ==
                "cyaxiverse-physical-spectrum-4"
            @test read(metadata(file)["kinetic_normalization"]) == "sugra_c1"
        end
        loaded = r.physical_spectrum(geom_idx)
        @test labels(loaded) == (:sugra_c1, false)
        @test loaded.m == spectrum.m
        @test loaded.schema_version == r.F1_SCHEMA_VERSION
        @test r.stored_kinetic_normalization(path; family=:f1) == (:sugra_c1, false)

        # A rewrite deletes the whole spectrum/physical group first; F2 and
        # F3 siblings are not touched.
        h5open(path, "r+") do file
            file["spectrum/physical/stale"] = [1.0]
            metadata(file)["stale"] = 1
            file["spectrum/masses/log10"] = [9.0]
            file["spectrum/axion_photon/identity/schema_version"] = 1
        end
        write_f1(path; quartics=false)
        h5open(path, "r") do file
            @test !haskey(file, "spectrum/physical/stale")
            @test !haskey(metadata(file), "stale")
            @test !haskey(file, "spectrum/physical/fpert_log10")
            @test read(metadata(file)["kinetic_normalization"]) == "sugra_c1"
            @test read(file["spectrum/masses/log10"]) == [9.0]
            @test read(file["spectrum/axion_photon/identity/schema_version"]) == 1
        end
    end

    # Inference: "-3" without a label.
    f1_fixture() do path
        h5open(path, "r+") do file
            set!("kinetic_normalization", nothing)(file)
            set!("schema_version", r.F1_LEGACY_SCHEMA_VERSION)(file)
        end
        @test labels(r.physical_spectrum(geom_idx)) == (:legacy_c_half, true)
        @test r.stored_kinetic_normalization(path; family=:f1) == (:legacy_c_half, true)
    end
    # Inference: unversioned complete layout without fpert_log10.
    f1_fixture(; quartics=false) do path
        h5open(path, "r+") do file
            set!("kinetic_normalization", nothing)(file)
            set!("schema_version", nothing)(file)
            @test !haskey(file, "spectrum/physical/fpert_log10")
        end
        loaded = r.physical_spectrum(geom_idx)
        @test labels(loaded) == (:legacy_c_half, true)
        @test loaded.schema_version === nothing
        @test loaded.fpert == Float64[]
        @test r.stored_kinetic_normalization(path; family=:f1) == (:legacy_c_half, true)
    end

    # R-007 rejections, through the reader and stored_kinetic_normalization.
    function f1_rejected(edits...)
        f1_fixture() do path
            h5open(file -> foreach(e -> e(file), edits), path, "r+")
            reader = try
                r.physical_spectrum(geom_idx)
                false
            catch err
                message = sprint(showerror, err)
                err isa ArgumentError && occursin(path, message) && occursin("F1", message)
            end
            stored = try
                r.stored_kinetic_normalization(path; family=:f1)
                false
            catch err
                err isa ArgumentError
            end
            reader && stored
        end
    end
    unversioned = (set!("schema_version", nothing), set!("kinetic_normalization", nothing))
    @test f1_rejected(set!("kinetic_normalization", nothing))
    @test f1_rejected(set!("kinetic_normalization", "bogus"))
    @test f1_rejected(set!("kinetic_normalization", "legacy_c_half"))
    @test f1_rejected(set!("kinetic_normalization", "a96_reproduction_raw_radian"))
    @test f1_rejected(set!("schema_version", "cyaxiverse-physical-spectrum-5"))
    @test f1_rejected(set!("schema_version", r.F1_LEGACY_SCHEMA_VERSION))
    @test f1_rejected(set!("schema_version", nothing))
    @test f1_rejected(unversioned..., set!("runtime_seconds", nothing))
    @test f1_rejected(unversioned...,
        file -> HDF5.delete_object(file["spectrum/physical"], "fK_log10"))

    # Summary CSV: label column, and no append onto a legacy summary.
    mktempdir() do dir
        summary = joinpath(dir, "summary.csv")
        _write_summary_header(summary)
        _append_summary(summary, geom_idx; status="success", prec=200,
            threshold_log10=-30.0, spectrum=spectrum, quartics=true)
        lines = readlines(summary)
        @test split(lines[1], ',')[end] == "kinetic_normalization"
        @test split(lines[2], ',')[end] == "sugra_c1"
        @test length(split(lines[2], ',')) == length(split(lines[1], ','))
        _write_summary_header(summary; append=true)
        @test length(readlines(summary)) == 2

        legacy_summary = joinpath(dir, "legacy.csv")
        write(legacy_summary, replace(SUMMARY_HEADER, ",kinetic_normalization" => "") * "\n")
        err = try
            _write_summary_header(legacy_summary; append=true)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("no kinetic_normalization column", sprint(showerror, err))
        @test length(readlines(legacy_summary)) == 1

        empty_summary = joinpath(dir, "empty.csv")
        touch(empty_summary)
        _write_summary_header(empty_summary; append=true)
        @test readlines(empty_summary) == [SUMMARY_HEADER]
    end
end

@testset "CYAX-0232 F3 identity 2, label and CSV (T032)" begin
    ap = CYAxiverse.axion_photon
    r = CYAxiverse.read
    fixture = first(r005_fixtures())
    identity_path = "spectrum/axion_photon/identity"
    set!(name, value) = identity -> begin
        haskey(identity, name) && HDF5.delete_object(identity, name)
        value === nothing || (identity[name] = value)
    end
    r005_with_geometry(fixture, 1) do geom_idx, path, root
        result = ap._run_local_scan(path)
        @test result.identity.schema_version == 2
        ap.write_axion_photon_result(path, result)
        h5open(path, "r") do file
            @test read(file[identity_path * "/schema_version"]) == 2
            @test read(file[identity_path * "/kinetic_normalization"]) == "sugra_c1"
        end
        restored = ap.read_axion_photon_result(path)
        @test restored.identity == result.identity
        @test r.stored_kinetic_normalization(path; family=:f3) == (:sugra_c1, false)
        csv = joinpath(root, "scan.csv")
        ap.write_scan_csv(csv, [restored])
        lines = readlines(csv)
        @test endswith(lines[1], ",kinetic_normalization,kinetic_normalization_inferred")
        @test endswith(lines[2], ",sugra_c1,false")
        @test length(split(lines[2], ',')) == length(split(lines[1], ','))

        # Legacy identity version 1 without a label: read by inference, and
        # the stored version is kept.
        h5open(path, "r+") do file
            identity = file[identity_path]
            set!("schema_version", 1)(identity)
            set!("kinetic_normalization", nothing)(identity)
        end
        legacy = ap.read_axion_photon_result(path)
        @test legacy.identity.schema_version == 1
        @test r.f3_identity_label(legacy.identity.schema_version) == (:legacy_c_half, true)
        @test r.stored_kinetic_normalization(path; family=:f3) == (:legacy_c_half, true)
        ap.write_scan_csv(csv, [legacy])
        @test endswith(readlines(csv)[2], ",legacy_c_half,true")

        # Mixed labels are not combined into one CSV.
        err = try
            ap.write_scan_csv(csv, [restored, legacy])
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("sugra_c1", sprint(showerror, err)) &&
            occursin("legacy_c_half", sprint(showerror, err))

        # The writer refuses a legacy result read back from the file, and a
        # current result does not overwrite the legacy group without force.
        @test_throws ArgumentError ap.write_axion_photon_result(path, legacy; force=true)
        @test_throws ArgumentError ap.write_axion_photon_result(path, result)
        @test r.stored_kinetic_normalization(path; family=:f3) == (:legacy_c_half, true)
        ap.write_axion_photon_result(path, result; force=true)
        @test r.stored_kinetic_normalization(path; family=:f3) == (:sugra_c1, false)
    end

    # R-007 rejections, through the reader and stored_kinetic_normalization.
    function f3_rejected(edits...)
        r005_with_geometry(fixture, 1) do geom_idx, path, _
            ap.write_axion_photon_result(path, ap._run_local_scan(path))
            h5open(file -> foreach(e -> e(file[identity_path]), edits), path, "r+")
            reader = try
                ap.read_axion_photon_result(path)
                false
            catch err
                message = sprint(showerror, err)
                err isa ArgumentError && occursin(path, message) && occursin("F3", message)
            end
            stored = try
                r.stored_kinetic_normalization(path; family=:f3)
                false
            catch err
                err isa ArgumentError
            end
            reader && stored
        end
    end
    @test f3_rejected(set!("kinetic_normalization", nothing))
    @test f3_rejected(set!("kinetic_normalization", "bogus"))
    @test f3_rejected(set!("kinetic_normalization", "legacy_c_half"))
    @test f3_rejected(set!("kinetic_normalization", "p96_reproduction_c_half"))
    @test f3_rejected(set!("schema_version", 1))
    @test f3_rejected(set!("schema_version", 3))
    @test f3_rejected(set!("schema_version", "2"))
end

# The F4 driver defines `main`; keep it in its own module.
module CYAX0232F4Driver
include(joinpath(@__DIR__, "..", "scripts", "fuzzy_axion_model_stage_driver.jl"))
end

@testset "CYAX-0232 F4 driver label and refusal (T033)" begin
    r = CYAxiverse.read
    driver = CYAX0232F4Driver
    # Synthetic record of scripts/test_fuzzy_axion_model_stage_driver.py,
    # written as h5py would write it (2D datasets transposed).
    Q = [1 0 1 0 2 1; 0 1 0 1 1 2]
    tau = [1.5, 2.0, 1.2, 1.8, 3.0, 2.5]
    Kinv = [2.0 0.3; 0.3 1.5]
    function write_input(path; schema="kaehler-point-provenance-2", label="sugra_c1")
        h5open(path, "w") do file
            file["record_count"] = 1
            HDF5.attributes(file)["record_metadata_schema"] = schema
            label === nothing || (file["kinetic_normalization"] = label)
            group = create_group(create_group(file, "records"), "0")
            group["Q"] = permutedims(Q)
            group["tau"] = tau
            group["cy_volume"] = 5.0
            group["inverse_metric"] = permutedims(Kinv)
        end
    end
    run_driver(input, output) = redirect_stdout(devnull) do
        driver.run_model_stage_driver(input, output, 0.5, complex(1.0, 0.0))
    end
    refused(f) = try
        f()
        false
    catch err
        err isa ArgumentError
    end

    mktempdir() do dir
        input = joinpath(dir, "input.h5")
        output = joinpath(dir, "output.h5")
        write_input(input)
        @test h5open(file -> r.f5_model_stage_input_label(file, input), input) ==
            (:sugra_c1, false)
        run_driver(input, output)
        h5open(output, "r") do file
            @test read(file["schema_version"]) == "cyaxiverse-fuzzy-axion-model-stage-1"
            @test read(file["kinetic_normalization"]) == "sugra_c1"
            @test read(file["total_model_count"]) == length(read(file["model_record_index"]))
        end
        @test r.stored_kinetic_normalization(output; family=:f4) == (:sugra_c1, false)
        # A current output may be replaced.
        run_driver(input, output)
        @test r.stored_kinetic_normalization(output; family=:f4) == (:sugra_c1, false)

        # A legacy (unversioned, unlabelled) output is read by inference and
        # never overwritten.
        h5open(output, "r+") do file
            HDF5.delete_object(file, "schema_version")
            HDF5.delete_object(file, "kinetic_normalization")
        end
        @test r.stored_kinetic_normalization(output; family=:f4) == (:legacy_c_half, true)
        @test refused(() -> run_driver(input, output))
        @test r.stored_kinetic_normalization(output; family=:f4) == (:legacy_c_half, true)
        # Nor is a file that is not an F4 output.
        other = joinpath(dir, "other.h5")
        write(other, "not hdf5")
        @test refused(() -> run_driver(input, other))
        @test read(other, String) == "not hdf5"

        # Inputs that are legacy, unlabelled or unknown are refused before any
        # output is written.
        fresh = joinpath(dir, "fresh.h5")
        for (schema, label) in (("kaehler-point-provenance-1", nothing),
                ("kaehler-point-provenance-1", "sugra_c1"),
                ("kaehler-point-provenance-2", nothing),
                ("kaehler-point-provenance-2", "legacy_c_half"),
                ("kaehler-point-provenance-3", "sugra_c1"))
            write_input(input; schema, label)
            @test refused(() -> run_driver(input, fresh))
            @test !isfile(fresh)
        end
        write_input(input; schema="kaehler-point-provenance-1", label=nothing)
        @test h5open(file -> r.f5_model_stage_input_label(file, input), input) ==
            (:legacy_c_half, true)
    end

    # R-007 rejections for F4 files, through stored_kinetic_normalization.
    function f4_rejected(datasets)
        mktempdir() do dir
            path = joinpath(dir, "f4.h5")
            h5open(path, "w") do file
                for (name, value) in datasets
                    file[name] = value
                end
            end
            try
                r.stored_kinetic_normalization(path; family=:f4)
                false
            catch err
                err isa ArgumentError && occursin(path, sprint(showerror, err))
            end
        end
    end
    legacy = ["total_model_count" => 3, "model_mass_reference_log10_ev" => [1.0]]
    version = "schema_version" => "cyaxiverse-fuzzy-axion-model-stage-1"
    @test f4_rejected([legacy; version])
    @test f4_rejected([legacy; version; "kinetic_normalization" => "legacy_c_half"])
    @test f4_rejected([legacy; version; "kinetic_normalization" => "bogus"])
    @test f4_rejected([legacy; "schema_version" => "cyaxiverse-fuzzy-axion-model-stage-2";
        "kinetic_normalization" => "sugra_c1"])
    @test f4_rejected([legacy; "kinetic_normalization" => "sugra_c1"])
    @test f4_rejected(["total_model_count" => 3])
end

const T035_INFL = only(f for f in r005_fixtures() if f.name == "infl")

@testset "CYAX-0232 F6 inflation group label (T035)" begin
    stored(path) = CYAxiverse.read.stored_kinetic_normalization(path; family=:f6)
    write_f6(geom_idx) = write_inflation_group!(geom_idx,
        (; status=:success, result=run_geometry(geom_idx; max_branches=1_000)),
        (; status=:not_run, message="T035: e-fold scan not run");
        catastrophe_settings=(; max_branches=1_000),
        efold_settings=(; precision_bits=128), force=true)
    set!(name, value) = group -> begin
        haskey(group, name) && HDF5.delete_object(group, name)
        value === nothing || (group[name] = value)
    end
    version(v) = "cyaxiverse-phase3-orientifold-inflation-" * v

    r005_with_geometry(T035_INFL, 1) do geom_idx, path, _
        write_f6(geom_idx)
        h5open(path, "r") do file
            @test read(file["inflation/schema_version"]) == version("2.4")
            @test read(file["inflation/kinetic_normalization"]) == "sugra_c1"
            @test read(file["inflation/catastrophes/legacy_v1/schema_version"]) ==
                version("1.0")
        end
        @test stored(path) == (:sugra_c1, false)
        # The nested legacy_v1 marker is not the group version.
        h5open(file -> set!("schema_version", version("2.3"))(
            file["inflation/catastrophes/legacy_v1"]), path, "r+")
        @test stored(path) == (:sugra_c1, false)
        # Every historical group version without a label is inferred.
        for v in ("2.3", "2.2", "2.1", "2.0", "1.0")
            h5open(path, "r+") do file
                set!("kinetic_normalization", nothing)(file["inflation"])
                set!("schema_version", version(v))(file["inflation"])
            end
            @test stored(path) == (:legacy_c_half, true)
        end
    end

    function f6_rejected(edits...)
        r005_with_geometry(T035_INFL, 1) do geom_idx, path, _
            write_f6(geom_idx)
            h5open(file -> foreach(e -> e(file["inflation"]), edits), path, "r+")
            try
                stored(path)
                false
            catch err
                message = sprint(showerror, err)
                err isa ArgumentError && occursin(path, message) && occursin("F6", message)
            end
        end
    end
    @test f6_rejected(set!("kinetic_normalization", nothing))
    @test f6_rejected(set!("kinetic_normalization", "legacy_c_half"))
    @test f6_rejected(set!("kinetic_normalization", "bogus"))
    @test f6_rejected(set!("schema_version", version("2.3")))
    @test f6_rejected(set!("schema_version", version("2.5")))
    @test f6_rejected(set!("schema_version", nothing))
end

@testset "CYAX-0232 inflation CSV labels (T036)" begin
    @test INFLATION_SCAN_CONTRACT_VERSION == "5"
    @test INFLATION_DIAGNOSTIC_SCHEMA_VERSION == "2"
    @test INFLATION_SHARD_SCHEMA_VERSION == "3"
    @test REFINEMENT_SCHEMA_VERSION == "4"
    @test FLOW_SCHEMA_VERSION == "2"
    @test PILOT_SCHEMA_VERSION == "7"
    @test INFLATION_PHYSICAL_KINETIC_NORMALIZATION == String(CYAxiverse.KINETIC_NORMALIZATION)
    for fields in (REFINEMENT_FIELDS, FLOW_FIELDS, PILOT_COMMON_FIELDS)
        @test fields[2:3] == (:schema_version, :kinetic_normalization)
    end
    @test :kinetic_normalization in SCAN_PREP_FIELDS
    @test :kinetic_normalization in INFLATION_SHARD_FIELDS
    function csv_rows(path)
        lines = readlines(path)
        header = _inflation_shard_csv_fields(lines[1])
        [Dict(zip(header, _inflation_shard_csv_fields(line))) for line in lines[2:end]
            if !isempty(strip(line))]
    end
    without_label(header) = replace(header, ",kinetic_normalization" => "")

    r005_with_geometry(T035_INFL, 1) do geom_idx, path, root
        row = run_geometry(geom_idx; max_branches=1_000)
        @test (row.contract_version, row.diagnostic_schema_version,
            row.kinetic_normalization) == ("5", "2", "sugra_c1")

        # Scan-prep summary: round trip, and no append onto a legacy header.
        summary = joinpath(root, "prep.csv")
        _scan_prep_write_header(summary)
        _scan_prep_append(summary, row; data_dir=root, max_branches=1_000)
        prep = only(csv_rows(summary))
        @test (prep["contract_version"], prep["diagnostic_schema_version"],
            prep["kinetic_normalization"]) == ("5", "2", "sugra_c1")
        legacy_summary = joinpath(root, "legacy_prep.csv")
        write(legacy_summary, without_label(SCAN_PREP_HEADER) * "\n")
        @test_throws ArgumentError _scan_prep_write_header(legacy_summary; append=true)

        # Shard rows and the merged CSV.
        shard = joinpath(root, "shards", "shard.csv")
        inflation_prepare_shard(shard)
        inflation_append_shard_row(shard, row; run_id="t036", shard_index=1,
            shard_count=1, attempt=1, started_unix_s=0.0, finished_unix_s=1.0,
            data_dir=root, max_branches=1_000)
        shard_row = only(csv_rows(shard))
        @test (shard_row["shard_schema_version"], shard_row["contract_version"],
            shard_row["kinetic_normalization"]) == ("3", "5", "sugra_c1")
        legacy_shard = joinpath(root, "legacy_shards", "shard.csv")
        mkpath(dirname(legacy_shard))
        write(legacy_shard, without_label(INFLATION_SHARD_HEADER) * "\n")
        @test_throws ArgumentError inflation_prepare_shard(legacy_shard; append=true)
        merged = inflation_merge_shards([shard], joinpath(root, "merged.csv"))
        @test only(csv_rows(merged))["kinetic_normalization"] == "sugra_c1"

        # Scale-continuation report and shard rows.
        report = joinpath(root, "pilot", "report.csv")
        pilot_shards = joinpath(root, "pilot", "shards")
        options = _pilot_parse_args(["--data-dir", root, "--geometry",
            "$(geom_idx.h11),$(geom_idx.polytope),$(geom_idx.frst)",
            "--scale-grid", "1.0", "--scale-status", "homotopy_only",
            "--report", report, "--shard-dir", pilot_shards, "--run-id", "t036"])
        redirect_stdout(devnull) do
            run_scale_continuation(options)
        end
        pilot_rows = vcat(csv_rows(report),
            [csv_rows(p) for p in readdir(pilot_shards; join=true)]...)
        @test !isempty(pilot_rows)
        @test all(r -> r["schema_version"] == "7" &&
            r["kinetic_normalization"] == "sugra_c1", pilot_rows)
        @test startswith(readlines(report)[1], "row_type,schema_version,kinetic_normalization,")
    end
end

# The Appendix B script defines `main` and `_parse_args`; keep it in its own
# module, and skip its global `set_theme!` so other plots are unaffected.
module CYAX0232AppendixB
using CairoMakie
Base.include(ex -> (Meta.isexpr(ex, :call) && ex.args[1] === :set_theme!) ? nothing : ex,
    @__MODULE__, joinpath(@__DIR__, "..", "scripts", "reproduce_appendix_b_spectra.jl"))
end

@testset "CYAX-0232 derived writers (T037)" begin
    # Appendix B quantiles from F1 outputs.
    appendix = CYAX0232AppendixB
    spectrum = CYAxiverse.structs.PhysicalAxionSpectrum(
        [1.0, 2.0], [0, 1], zeros(2, 2), [1, -1], [3.0, 4.0],
        reshape([1, 1, 1, 2], 4, 1), [1], [0.5],
        reshape([1, 1, 2, 2], 4, 1), [-1], [0.7],
        -30.0, 200)
    function f1_files(root, versions)
        files = Tuple{Int,String}[]
        for (index, version) in enumerate(versions)
            geom = CYAxiverse.structs.GeometryIndex(2, index, 1)
            path = joinpath(root, "h11_002", "np_$(lpad(index, 7, '0'))", "cy_0000001", "cyax.h5")
            mkpath(dirname(path))
            h5open(file -> create_group(file, "cytools"), path, "cw")
            _write_result(path, geom, spectrum; prec=200, threshold_log10=-30.0,
                quartics=true, runtime_seconds=0.1, provisional=false, fK=[5.0, 6.0])
            h5open(path, "r+") do file
                metadata = file["spectrum/physical/metadata"]
                if version == :legacy
                    HDF5.delete_object(metadata, "kinetic_normalization")
                    HDF5.delete_object(metadata, "schema_version")
                    metadata["schema_version"] = CYAxiverse.read.F1_LEGACY_SCHEMA_VERSION
                elseif version == :unlabelled
                    HDF5.delete_object(metadata, "kinetic_normalization")
                end
            end
            push!(files, (2, path))
        end
        files
    end
    function summary_rows(root, values)
        masses, fpert, fK, completed, label = values
        path = joinpath(root, "appendix_b_quantiles.csv")
        appendix._write_summary(path, [2], masses, fpert, fK, completed, label)
        lines = readlines(path)
        header = split(lines[1], ',')
        [Dict(zip(header, split(line, ','))) for line in lines[2:end]]
    end
    mktempdir() do root
        values = appendix._read_values(f1_files(root, (:current, :current)))
        @test values[5] == (:sugra_c1, false)
        row = only(summary_rows(root, values))
        @test (row["geometries"], row["kinetic_normalization"],
            row["kinetic_normalization_inferred"]) == ("2", "sugra_c1", "false")
    end
    mktempdir() do root
        values = appendix._read_values(f1_files(root, (:legacy, :legacy)))
        @test values[5] == (:legacy_c_half, true)
        row = only(summary_rows(root, values))
        @test (row["kinetic_normalization"], row["kinetic_normalization_inferred"]) ==
            ("legacy_c_half", "true")
    end
    mktempdir() do root
        files = f1_files(root, (:current, :legacy))
        err = try
            appendix._read_values(files)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("sugra_c1", sprint(showerror, err)) &&
            occursin("legacy_c_half", sprint(showerror, err))
    end
    mktempdir() do root
        # An R-007 rejection is not skipped.
        files = f1_files(root, (:current, :unlabelled))
        @test_throws ArgumentError appendix._read_values(files)
        # A file without spectrum/physical/m is still skipped.
        h5open(file -> HDF5.delete_object(file, "spectrum/physical/m"), files[2][2], "r+")
        @test appendix._read_values(files)[4] == Dict(2 => 1)
    end

    # Inflation shard merge and pilot report.
    function shard_with_label(dir, label; frst=1)
        path = joinpath(dir, "shard_$(frst).csv")
        inflation_prepare_shard(path)
        row = (; contract_version=INFLATION_SCAN_CONTRACT_VERSION,
            diagnostic_schema_version=INFLATION_DIAGNOSTIC_SCHEMA_VERSION,
            kinetic_normalization=label, h11=2, polytope=1, frst, status=:success,
            instantons=3, strong_hierarchy=true, leading_log_gap=1.0,
            log_scale_span=2.0, branch_count=1, candidate_slowroll_saddles=0,
            total_seconds=0.1, stage_allocated_bytes=10, stage_output_bytes=5)
        inflation_append_shard_row(path, row; run_id="t037", shard_index=1,
            shard_count=1, attempt=1, started_unix_s=0.0, finished_unix_s=1.0,
            data_dir=dir, max_branches=10)
        path
    end
    mktempdir() do root
        current = shard_with_label(joinpath(root, "a"), "sugra_c1"; frst=1)
        other = shard_with_label(joinpath(root, "b"), "legacy_c_half"; frst=2)
        unlabelled = shard_with_label(joinpath(root, "c"), ""; frst=3)
        merged = joinpath(root, "merged.csv")
        err = try
            inflation_merge_shards([current, other], merged)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("sugra_c1", sprint(showerror, err)) &&
            occursin("legacy_c_half", sprint(showerror, err))
        @test !isfile(merged)
        @test_throws ArgumentError inflation_merge_shards([current, unlabelled], merged)
        @test !isfile(merged)

        rows = _inflation_pilot_terminal_rows(joinpath(root, "a"))
        @test only(rows).kinetic_normalization == "sugra_c1"
        reports = _inflation_pilot_report_rows(rows)
        @test all(report.kinetic_normalization == "sugra_c1" for report in reports)
        report_path = _inflation_pilot_write_report(joinpath(root, "report.csv"), reports)
        @test endswith(readlines(report_path)[1], ",kinetic_normalization")
        @test endswith(readlines(report_path)[2], ",sugra_c1")
        # A current-layout shard row labelled "legacy_c_half" fails the row
        # check (T063); mixed labels are rejected by the report.
        @test_throws ArgumentError _inflation_pilot_terminal_rows(joinpath(root, "b"))
        mixed = vcat(rows, [merge(only(rows), (; kinetic_normalization="legacy_c_half"))])
        @test_throws ArgumentError _inflation_pilot_report_rows(mixed)
        @test_throws ArgumentError _inflation_pilot_report_rows(
            _inflation_pilot_terminal_rows(joinpath(root, "c")))
    end
end

# CYAX-0232 T060: the R-007 inference and rejection matrix for the HDF5
# families F1–F4 and F6, one case per row, on minimal synthetic layouts and
# through `stored_kinetic_normalization`. F5 is covered by the Python tests
# (T061). Every rejection must be an ArgumentError naming the file.
const T060_F1_CURRENT = "cyaxiverse-physical-spectrum-4"
const T060_F1_LEGACY = "cyaxiverse-physical-spectrum-3"
const T060_F6 = "cyaxiverse-phase3-orientifold-inflation-"

function t060_build(family, file; version=nothing, label=nothing, extra=Dict())
    put!(group, name, value) = value === nothing || (group[name] = value)
    if family === :f1
        physical = create_group(create_group(file, "spectrum"), "physical")
        metadata = create_group(physical, "metadata")
        for name in ("m", "fK_log10")
            get(extra, "drop", "") == name || (physical[name] = [1.0])
        end
        physical["mode_indices"] = [1]
        for name in ("threshold_log10", "prec", "provisional", "runtime_seconds")
            get(extra, "drop", "") == name || (metadata[name] = 1)
        end
        put!(metadata, "schema_version", version)
        put!(metadata, "kinetic_normalization", label)
    elseif family === :f2
        spectrum = create_group(file, "spectrum")
        get(extra, "legacy_masses", true) && (spectrum["masses/log10"] = [1.0])
        if get(extra, "metadata", false)
            metadata = create_group(spectrum, "metadata")
            put!(metadata, "schema_version", version)
            put!(metadata, "kinetic_normalization", label)
            put!(metadata, "cubic_basis", get(extra, "cubic_basis", nothing))
        end
    elseif family === :f3
        identity = create_group(create_group(create_group(file, "spectrum"),
            "axion_photon"), "identity")
        put!(identity, "schema_version", version)
        put!(identity, "kinetic_normalization", label)
    elseif family === :f4
        get(extra, "drop", "") == "total_model_count" || (file["total_model_count"] = 1)
        file["model_mass_reference_log10_ev"] = [1.0]
        put!(file, "schema_version", version)
        put!(file, "kinetic_normalization", label)
    elseif family === :f6
        inflation = create_group(file, "inflation")
        put!(inflation, "schema_version", version)
        put!(inflation, "kinetic_normalization", label)
        inflation["catastrophes/legacy_v1/schema_version"] = T060_F6 * "1.0"
    end
end

function t060_case(family; expect, kwargs...)
    mktempdir() do dir
        path = joinpath(dir, "t060.h5")
        h5open(file -> t060_build(family, file; kwargs...), path, "w")
        if expect === :reject
            try
                CYAxiverse.read.stored_kinetic_normalization(path; family)
                false
            catch err
                err isa ArgumentError && occursin(path, sprint(showerror, err))
            end
        else
            CYAxiverse.read.stored_kinetic_normalization(path; family) == expect
        end
    end
end

@testset "CYAX-0232 R-007 inference and rejection matrix (T060)" begin
    current = (:sugra_c1, false)
    legacy = (:legacy_c_half, true)
    @testset "F1" begin
        @test t060_case(:f1; expect=current, version=T060_F1_CURRENT, label="sugra_c1")
        @test t060_case(:f1; expect=legacy, version=T060_F1_LEGACY)
        @test t060_case(:f1; expect=legacy)                                   # unversioned, complete
        @test t060_case(:f1; expect=:reject, version=T060_F1_CURRENT)         # current, no label
        @test t060_case(:f1; expect=:reject, version=T060_F1_CURRENT, label="bogus")
        @test t060_case(:f1; expect=:reject, version=T060_F1_CURRENT, label="legacy_c_half")
        @test t060_case(:f1; expect=:reject, version=T060_F1_CURRENT, label="p96_reproduction_c_half")
        @test t060_case(:f1; expect=:reject, version=T060_F1_LEGACY, label="sugra_c1")
        @test t060_case(:f1; expect=:reject, label="sugra_c1")                # unversioned, labelled
        @test t060_case(:f1; expect=:reject, extra=Dict("drop" => "runtime_seconds"))
        @test t060_case(:f1; expect=:reject, extra=Dict("drop" => "fK_log10"))
        @test t060_case(:f1; expect=:reject, version="cyaxiverse-physical-spectrum-5",
            label="sugra_c1")
    end
    @testset "F2" begin
        meta = Dict("metadata" => true, "cubic_basis" => "charge_basis")
        version = "cyaxiverse-spectrum-groups-1"
        @test t060_case(:f2; expect=current, version, label="sugra_c1", extra=meta)
        @test t060_case(:f2; expect=current, version, label="sugra_c1",
            extra=Dict("metadata" => true, "cubic_basis" => "canonical_mass_basis"))
        @test t060_case(:f2; expect=legacy)
        @test t060_case(:f2; expect=:reject, version, extra=meta)
        @test t060_case(:f2; expect=:reject, version, label="bogus", extra=meta)
        @test t060_case(:f2; expect=:reject, version, label="legacy_c_half", extra=meta)
        @test t060_case(:f2; expect=:reject, version, label="a96_reproduction_raw_radian", extra=meta)
        @test t060_case(:f2; expect=:reject, version, label="sugra_c1",
            extra=Dict("metadata" => true))                                   # no cubic_basis
        @test t060_case(:f2; expect=:reject, version, label="sugra_c1",
            extra=Dict("metadata" => true, "cubic_basis" => "mass_basis"))
        @test t060_case(:f2; expect=:reject, label="sugra_c1", extra=meta)     # no version
        @test t060_case(:f2; expect=:reject, version="cyaxiverse-spectrum-groups-2",
            label="sugra_c1", extra=meta)
        @test t060_case(:f2; expect=:reject, extra=Dict("legacy_masses" => false))
    end
    @testset "F3" begin
        @test t060_case(:f3; expect=current, version=2, label="sugra_c1")
        @test t060_case(:f3; expect=legacy, version=1)
        @test t060_case(:f3; expect=:reject, version=2)
        @test t060_case(:f3; expect=:reject, version=2, label="bogus")
        @test t060_case(:f3; expect=:reject, version=2, label="legacy_c_half")
        @test t060_case(:f3; expect=:reject, version=2, label="p96_reproduction_c_half")
        @test t060_case(:f3; expect=:reject, version=1, label="sugra_c1")
        @test t060_case(:f3; expect=:reject, version=3, label="sugra_c1")
        @test t060_case(:f3; expect=:reject, version="2", label="sugra_c1")
        @test t060_case(:f3; expect=:reject)                                   # no version
    end
    @testset "F4" begin
        version = "cyaxiverse-fuzzy-axion-model-stage-1"
        @test t060_case(:f4; expect=current, version, label="sugra_c1")
        @test t060_case(:f4; expect=legacy)
        @test t060_case(:f4; expect=:reject, version)
        @test t060_case(:f4; expect=:reject, version, label="bogus")
        @test t060_case(:f4; expect=:reject, version, label="legacy_c_half")
        @test t060_case(:f4; expect=:reject, version, label="a96_reproduction_raw_radian")
        @test t060_case(:f4; expect=:reject, label="sugra_c1")
        @test t060_case(:f4; expect=:reject, extra=Dict("drop" => "total_model_count"))
        @test t060_case(:f4; expect=:reject, version="cyaxiverse-fuzzy-axion-model-stage-2",
            label="sugra_c1")
    end
    @testset "F6" begin
        @test t060_case(:f6; expect=current, version=T060_F6 * "2.4", label="sugra_c1")
        for v in ("2.3", "2.2", "2.1", "2.0", "1.0")
            @test t060_case(:f6; expect=legacy, version=T060_F6 * v)
            @test t060_case(:f6; expect=:reject, version=T060_F6 * v, label="sugra_c1")
        end
        @test t060_case(:f6; expect=:reject, version=T060_F6 * "2.4")
        @test t060_case(:f6; expect=:reject, version=T060_F6 * "2.4", label="bogus")
        @test t060_case(:f6; expect=:reject, version=T060_F6 * "2.4", label="legacy_c_half")
        @test t060_case(:f6; expect=:reject, version=T060_F6 * "2.4", label="p96_reproduction_c_half")
        @test t060_case(:f6; expect=:reject, version=T060_F6 * "2.5", label="sugra_c1")
        @test t060_case(:f6; expect=:reject)
    end
end

# The statistics notebook's `read_geometry_row`, evaluated on its own.
module CYAX0232Notebook
using CYAxiverse, HDF5
let source = read(joinpath(@__DIR__, "..", "notebooks", "orientifold_axiverse_statistics.jl"), String)
    first_line = findfirst("    function check_pooled_labels(", source)[1]
    last_line = findnext("\n    end\n", source,
        findfirst("    function read_geometry_row(", source)[1])[end]
    Base.include_string(@__MODULE__, source[first_line:last_line])
end
end

@testset "CYAX-0232 R-007 errors propagate (T062)" begin
    ap = CYAxiverse.axion_photon
    fixture = first(r005_fixtures())
    identity_path = "spectrum/axion_photon/identity"
    set!(name, value) = identity -> begin
        haskey(identity, name) && HDF5.delete_object(identity, name)
        value === nothing || (identity[name] = value)
    end
    r005_with_geometry(fixture, 1) do geom_idx, path, _
        result = ap._run_local_scan(path)
        ap.write_axion_photon_result(path, result)
        @test ap._has_axion_photon(path)
        # Legacy identity 1: present, not absent.
        h5open(path, "r+") do file
            set!("schema_version", 1)(file[identity_path])
            set!("kinetic_normalization", nothing)(file[identity_path])
        end
        @test ap._has_axion_photon(path)
        @test h5open(file -> ap._validate_axion_photon_group(file["spectrum/axion_photon"]),
            path, "r")
        @test_throws ArgumentError ap.write_axion_photon_result(path, result)
        # Identity 2 without a label: every site raises instead of "absent".
        h5open(file -> set!("schema_version", 2)(file[identity_path]), path, "r+")
        @test_throws ArgumentError ap._has_axion_photon(path)
        @test_throws ArgumentError h5open(file -> ap._has_axion_photon(file), path, "r")
        @test_throws ArgumentError h5open(
            file -> ap._validate_axion_photon_group(file["spectrum/axion_photon"]), path, "r")
        @test_throws ArgumentError ap.write_axion_photon_result(path, result)
        @test_throws ArgumentError ap.read_axion_photon_result(path)
        # The invalid group was not overwritten.
        @test h5open(file -> !haskey(file[identity_path], "kinetic_normalization"), path, "r")
    end

    # Notebook reader: label errors propagate; other defects still exclude.
    spectrum = CYAxiverse.structs.PhysicalAxionSpectrum(
        [1.0, 2.0], [0, 1], zeros(2, 2), [1, -1], [3.0, 4.0],
        reshape([1, 1, 1, 2], 4, 1), [1], [0.5],
        reshape([1, 1, 2, 2], 4, 1), [-1], [0.7],
        -30.0, 200)
    mktempdir() do root
        path = joinpath(root, "h11_002", "np_0000001", "cy_0000001", "cyax.h5")
        mkpath(dirname(path))
        h5open(file -> create_group(file, "cytools"), path, "cw")
        _write_result(path, CYAxiverse.structs.GeometryIndex(2, 1, 1), spectrum;
            prec=200, threshold_log10=-30.0, quartics=true, runtime_seconds=0.1,
            provisional=false, fK=[5.0, 6.0])
        row = withenv("CYAXIVERSE_DATA_DIR" => root) do
            CYAX0232Notebook.read_geometry_row(root, 2, 1, 1)
        end
        @test row.status == "excluded"                    # no vacua: an ordinary exclusion
        h5open(file -> HDF5.delete_object(file["spectrum/physical/metadata"],
            "kinetic_normalization"), path, "r+")
        @test_throws ArgumentError withenv("CYAXIVERSE_DATA_DIR" => root) do
            CYAX0232Notebook.read_geometry_row(root, 2, 1, 1)
        end
        h5open(path, "r+") do file
            file["spectrum/physical/metadata/kinetic_normalization"] = "sugra_c1"
            file["inflation/schema_version"] = "cyaxiverse-phase3-orientifold-inflation-2.4"
        end
        @test_throws ArgumentError withenv("CYAXIVERSE_DATA_DIR" => root) do
            CYAX0232Notebook.read_geometry_row(root, 2, 1, 1)
        end
    end
end

@testset "CYAX-0232 inflation CSV row labels (T063)" begin
    label(header, values; path="f.csv", row=2) = inflation_csv_row_label(header,
        [get(values, column, "x") for column in header]; path, row)
    rejected(header, values) = try
        label(header, values)
        false
    catch err
        err isa ArgumentError && occursin("f.csv, row 2", sprint(showerror, err))
    end
    # The current version values the reader expects are the writers' constants.
    @test INFLATION_CSV_CURRENT_VERSIONS == Dict(
        "contract_version" => INFLATION_SCAN_CONTRACT_VERSION,
        "diagnostic_schema_version" => INFLATION_DIAGNOSTIC_SCHEMA_VERSION,
        "shard_schema_version" => INFLATION_SHARD_SCHEMA_VERSION)
    @test INFLATION_CSV_CURRENT_SCHEMA_BY_ROW_TYPE == Dict(
        "point" => REFINEMENT_SCHEMA_VERSION, "summary" => REFINEMENT_SCHEMA_VERSION,
        "flow" => FLOW_SCHEMA_VERSION, "scale" => PILOT_SCHEMA_VERSION,
        "branch" => PILOT_SCHEMA_VERSION)

    # One test per recorded legacy entry (A3.4), and a contradictory version
    # value under each recorded header.
    @test length(INFLATION_CSV_LEGACY_ENTRIES) == 14
    for entry in INFLATION_CSV_LEGACY_ENTRIES
        header = INFLATION_CSV_LEGACY_HEADERS[entry.header]
        values = Dict{String, String}(entry.versions)
        entry.row_types === nothing || (values["row_type"] = first(entry.row_types))
        @test label(header, values) == ("legacy_c_half", true)
        for column in keys(entry.versions)
            @test rejected(header, merge(values, Dict(column => "99")))
        end
        entry.row_types === nothing ||
            @test rejected(header, merge(values, Dict("row_type" => "unknown")))
    end
    # Unknown keys: a recorded header with one extra or one missing column.
    prep = INFLATION_CSV_LEGACY_HEADERS["PREP_C"]
    prep_values = Dict("contract_version" => "4", "diagnostic_schema_version" => "1")
    @test rejected([prep; "unrecorded_column"], prep_values)
    @test rejected(prep[1:end-1], prep_values)
    @test_throws ArgumentError inflation_csv_row_label(prep, ["4", "1"]; path="f.csv", row=2)

    # Current layouts.
    scan = string.(SCAN_PREP_FIELDS)
    current = Dict("contract_version" => "5", "diagnostic_schema_version" => "2",
        "kinetic_normalization" => "sugra_c1")
    @test label(scan, current) == ("sugra_c1", false)
    @test label(scan, merge(current, Dict("kinetic_normalization" =>
        "a96_reproduction_raw_radian"))) == ("a96_reproduction_raw_radian", false)
    for bad in (Dict("kinetic_normalization" => ""),
            Dict("kinetic_normalization" => "legacy_c_half"),
            Dict("kinetic_normalization" => "bogus"),
            Dict("contract_version" => "4"), Dict("diagnostic_schema_version" => "1"))
        @test rejected(scan, merge(current, bad))
    end
    shard = string.(INFLATION_SHARD_FIELDS)
    @test label(shard, merge(current, Dict("shard_schema_version" => "3"))) == ("sugra_c1", false)
    @test rejected(shard, merge(current, Dict("shard_schema_version" => "2")))
    for (fields, row_type, version) in ((PILOT_COMMON_FIELDS, "scale", "7"),
            (PILOT_COMMON_FIELDS, "branch", "7"), (REFINEMENT_FIELDS, "point", "4"),
            (REFINEMENT_FIELDS, "summary", "4"), (FLOW_FIELDS, "flow", "2"))
        values = Dict("row_type" => row_type, "schema_version" => version,
            "kinetic_normalization" => "sugra_c1")
        @test label(string.(fields), values) == ("sugra_c1", false)
        @test rejected(string.(fields), merge(values, Dict("schema_version" => "3")))
        @test rejected(string.(fields), merge(values,
            Dict("kinetic_normalization" => "a96_reproduction_raw_radian")))
    end
    @test rejected(string.(PILOT_COMMON_FIELDS), Dict("row_type" => "other",
        "schema_version" => "7", "kinetic_normalization" => "sugra_c1"))

    # The resume readers let these errors propagate (T062).
    mktempdir() do dir
        summary = joinpath(dir, "prep.csv")
        _scan_prep_write_header(summary)
        open(summary, "a") do io
            values = merge(Dict(string(f) => "1" for f in SCAN_PREP_FIELDS),
                Dict("contract_version" => "5", "diagnostic_schema_version" => "2",
                    "kinetic_normalization" => "", "data_dir" => dir, "status" => "success",
                    "negative_mode_range" => "all"))
            println(io, join([values[string(f)] for f in SCAN_PREP_FIELDS], ','))
        end
        @test_throws ArgumentError _scan_prep_completed(summary; data_dir=dir, max_branches=1)
        shard = joinpath(dir, "shards", "shard.csv")
        inflation_prepare_shard(shard)
        inflation_append_shard_row(shard, (; contract_version="5",
            diagnostic_schema_version="2", kinetic_normalization="", h11=2, polytope=1,
            frst=1, status=:success); run_id="t063", shard_index=1, shard_count=1,
            attempt=1, started_unix_s=0.0, finished_unix_s=1.0, data_dir=dir, max_branches=1)
        @test_throws ArgumentError inflation_completed_shard_geometries(
            dirname(shard); data_dir=dir, max_branches=1)
    end
end

@testset "CYAX-0232 combination rule (T064)" begin
    # Every combiner in the reader inventory rejects mixed labels with an
    # ArgumentError naming both; the others are tested with their writers:
    # write_scan_csv (T032), appendix B, inflation_merge_shards and the pilot
    # report (T037), the gap analysis and both shard merges (Python, T037/T061).
    row(f1, f6) = (; kinetic_normalization_f1=f1, kinetic_normalization_f6=f6)
    rows = [row((:sugra_c1, false), (:sugra_c1, false)),
        row((:sugra_c1, false), (:sugra_c1, false))]
    @test CYAX0232Notebook.check_pooled_labels(rows) === rows
    for mixed in ([rows[1], row((:legacy_c_half, true), (:sugra_c1, false))],
            [rows[1], row((:sugra_c1, false), (:legacy_c_half, true))])
        err = try
            CYAX0232Notebook.check_pooled_labels(mixed)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("sugra_c1", sprint(showerror, err)) &&
            occursin("legacy_c_half", sprint(showerror, err))
    end
end

@testset "CYAX-0232 batch resume (T065)" begin
    fixture = first(r005_fixtures())

    # F1: run_batch.
    r005_with_geometry(fixture, 1) do geom_idx, path, root
        summary = joinpath(root, "summary.csv")
        options(force) = (data_dir=root, h11=nothing, limit=nothing, offset=0,
            geometries=[geom_idx], prec=200, threshold_log10=nothing, quartics=false,
            force, summary, append_summary=false, hilbert=false)
        run(force) = redirect_stdout(devnull) do
            run_batch(options(force))
        end
        last_row() = Dict(zip(split(readlines(summary)[1], ','),
            split(readlines(summary)[end], ',')))
        @test run(false)
        @test run(false)                                   # complete: skipped
        @test last_row()["status"] == "skipped"
        h5open(path, "r+") do file
            metadata = file["spectrum/physical/metadata"]
            HDF5.delete_object(metadata, "kinetic_normalization")
            HDF5.delete_object(metadata, "schema_version")
            metadata["schema_version"] = "cyaxiverse-physical-spectrum-3"
        end
        @test !run(false)                                  # legacy, no --force
        @test last_row()["status"] == "failed"
        @test occursin("legacy_c_half", readlines(summary)[end])
        @test CYAxiverse.read.stored_kinetic_normalization(path; family=:f1) ==
            (:legacy_c_half, true)
        @test run(true)                                    # --force rewrites
        @test CYAxiverse.read.stored_kinetic_normalization(path; family=:f1) ==
            (:sugra_c1, false)
        h5open(file -> HDF5.delete_object(file["spectrum/physical/metadata"],
            "kinetic_normalization"), path, "r+")
        @test !run(false)                                  # invalid: failed, no crash
        @test last_row()["status"] == "failed"
    end

    # F3: run_batch_axion_photon.
    ap = CYAxiverse.axion_photon
    identity_path = "spectrum/axion_photon/identity"
    r005_with_geometry(fixture, 1) do geom_idx, path, root
        batch(force) = only(ap.run_batch_axion_photon(data_dir=root, h11s=(fixture.h11,),
            limit_per_h11=10, require_complete=false, force=force))
        @test batch(false).status == :written
        @test batch(false).status == :skipped
        h5open(path, "r+") do file
            identity = file[identity_path]
            HDF5.delete_object(identity, "schema_version")
            identity["schema_version"] = 1
            HDF5.delete_object(identity, "kinetic_normalization")
        end
        legacy = batch(false)
        @test legacy.status == :failed && occursin("legacy_c_half", legacy.error)
        @test batch(true).status == :written
        @test CYAxiverse.read.stored_kinetic_normalization(path; family=:f3) ==
            (:sugra_c1, false)
        h5open(file -> HDF5.delete_object(file[identity_path], "kinetic_normalization"),
            path, "r+")
        invalid = batch(false)
        @test invalid.status == :failed && occursin("kinetic_normalization", invalid.error)
    end

    # F6: _has_inflation_group and run_pipeline2's :blocked message.
    r005_with_geometry(T035_INFL, 1) do geom_idx, path, root
        write_inflation_group!(geom_idx,
            (; status=:success, result=run_geometry(geom_idx; max_branches=1_000)),
            (; status=:not_run, message="T065"); catastrophe_settings=(; max_branches=1_000),
            efold_settings=(; precision_bits=128), force=true,
            pipeline_config_digest="t065")
        @test _existing_inflation_label(path) == (:sugra_c1, false)
        h5open(path, "r+") do file
            inflation = file["inflation"]
            HDF5.delete_object(inflation, "kinetic_normalization")
            HDF5.delete_object(inflation, "schema_version")
            inflation["schema_version"] = "cyaxiverse-phase3-orientifold-inflation-2.3"
        end
        @test _existing_inflation_label(path) == (:legacy_c_half, true)
        @test !_has_inflation_group(path)
        blocked = redirect_stdout(devnull) do
            only(run_pipeline2(root; h11=T035_INFL.h11))
        end
        @test blocked.vacua_status == :blocked
        @test occursin("legacy_c_half", blocked.vacua_error)
        @test _phase3_cli_data_dir(["--force", root]) == _phase3_cli_data_dir([root])
    end

    # Scan-prep: legacy rows are not complete; no append onto a legacy summary.
    mktempdir() do dir
        header = INFLATION_CSV_LEGACY_HEADERS["PREP_C"]
        values = merge(Dict(c => "1" for c in header), Dict("contract_version" => "4",
            "diagnostic_schema_version" => "1", "data_dir" => dir, "status" => "success",
            "negative_mode_range" => "all", "max_branches" => "1"))
        legacy = joinpath(dir, "legacy.csv")
        write(legacy, join(header, ',') * "\n" * join([values[c] for c in header], ',') * "\n")
        @test isempty(_scan_prep_completed(legacy; data_dir=dir, max_branches=1))
        @test_throws ArgumentError _scan_prep_write_header(legacy; append=true)
    end
end

@testset "CYAX-0232 scale-continuation resume (T066)" begin
    r005_with_geometry(T035_INFL, 1) do geom_idx, path, root
        report = joinpath(root, "pilot", "report.csv")
        shards = joinpath(root, "pilot", "shards")
        args(extra...) = _pilot_parse_args(["--data-dir", root, "--geometry",
            "$(geom_idx.h11),$(geom_idx.polytope),$(geom_idx.frst)",
            "--scale-grid", "1.0", "--scale-status", "homotopy_only",
            "--report", report, "--shard-dir", shards, "--run-id", "t066", extra...])
        redirect_stdout(devnull) do
            run_scale_continuation(args())
        end
        completed = _pilot_completed_scales(report)
        @test (geom_idx.h11, geom_idx.polytope, geom_idx.frst, 1.0) in completed
        redirect_stdout(devnull) do                       # current files resume
            run_scale_continuation(args("--resume"))
        end

        # A legacy report (pilot 6) is refused by label, before the header check.
        legacy_header = INFLATION_CSV_LEGACY_HEADERS["PILOT_6"]
        legacy_row = [c == "row_type" ? "scale" : c == "schema_version" ? "6" :
            c in ("h11", "polytope", "frst") ? "1" : c == "sampled_scale" ? "1.0" : ""
            for c in legacy_header]
        cp(report, report * ".current"; force=true)
        write(report, join(legacy_header, ',') * "\n" * join(legacy_row, ',') * "\n")
        err = try
            run_scale_continuation(args("--resume"))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("legacy_c_half", sprint(showerror, err))
        @test !occursin("schema mismatch", sprint(showerror, err))
        write(report, join(legacy_header, ',') * "\n")       # header-only legacy file
        @test_throws ArgumentError _pilot_resume_label_check(report)
        @test isempty(_pilot_completed_scales(report))
        write(report, join(legacy_header, ',') * "\n" * join(legacy_row, ',') * "\n")
        @test isempty(_pilot_completed_scales(report))       # legacy rows: not complete

        # The three former silent skips now raise.
        current = readlines(report * ".current")
        header = current[1]
        write(report, replace(header, "sampled_scale" => "other_scale") * "\n" * current[2] * "\n")
        @test_throws ArgumentError _pilot_completed_scales(report)   # missing column
        write(report, header * "\n" * current[2] * ",extra\n")
        @test_throws ArgumentError _pilot_completed_scales(report)   # field count
        columns = _pilot_csv_fields(header)
        fields = _pilot_csv_fields(current[2])
        fields[findfirst(==("h11"), columns)] = "not-an-int"
        write(report, header * "\n" * join(fields, ',') * "\n")
        @test_throws ArgumentError _pilot_completed_scales(report)   # parse error
    end
end

# The legacy-spectrum migration defines `main`; keep it in its own module.
module CYAX0232Migrate
include(joinpath(@__DIR__, "..", "scripts", "migrate_legacy_spectra.jl"))
end

@testset "CYAX-0232 layout migrations (T067)" begin
    r = CYAxiverse.read
    function legacy_source(path; version=nothing)
        h5open(path, "w") do file
            physical = create_group(create_group(create_group(file, "cytools"),
                "spectrum"), "physical")
            physical["m"] = [1.0]
            physical["mode_indices"] = [1]
            physical["fK_log10"] = [2.0]
            metadata = create_group(file, "metadata")
            for name in ("threshold_log10", "prec", "provisional", "runtime_seconds")
                metadata[name] = 1
            end
            version === nothing || (metadata["schema_version"] = version)
        end
        path
    end
    mktempdir() do dir
        source = legacy_source(joinpath(dir, "legacy.h5"))
        target = joinpath(dir, "cyax.h5")
        h5open(file -> create_group(file, "cytools"), target, "w")
        @test CYAX0232Migrate.migrate_file!(source, target)
        h5open(target, "r") do file
            metadata = file["spectrum/physical/metadata"]
            @test !haskey(metadata, "kinetic_normalization")      # no label added
            @test !haskey(metadata, "schema_version")             # no version added
        end
        @test r.stored_kinetic_normalization(target; family=:f1) == (:legacy_c_half, true)
        @test !CYAX0232Migrate.migrate_file!(source, target)      # legacy target: skipped

        # A versioned source keeps its version.
        versioned = legacy_source(joinpath(dir, "legacy3.h5");
            version="cyaxiverse-physical-spectrum-3")
        target3 = joinpath(dir, "cyax3.h5")
        h5open(file -> create_group(file, "cytools"), target3, "w")
        @test CYAX0232Migrate.migrate_file!(versioned, target3)
        @test h5open(file -> read(file["spectrum/physical/metadata/schema_version"]),
            target3) == "cyaxiverse-physical-spectrum-3"

        # A current-version target is refused.
        current = joinpath(dir, "cyax4.h5")
        h5open(file -> create_group(file, "cytools"), current, "w")
        spectrum = CYAxiverse.structs.PhysicalAxionSpectrum(
            [1.0, 2.0], [0, 1], zeros(2, 2), [1, -1], [3.0, 4.0],
            reshape([1, 1, 1, 2], 4, 1), [1], [0.5],
            reshape([1, 1, 2, 2], 4, 1), [-1], [0.7], -30.0, 200)
        _write_result(current, CYAxiverse.structs.GeometryIndex(2, 1, 1), spectrum;
            prec=200, threshold_log10=-30.0, quartics=false, runtime_seconds=0.1,
            provisional=false, fK=[5.0, 6.0])
        @test_throws ArgumentError CYAX0232Migrate.migrate_file!(source, current)

        # Quartic-index migration: a current-version F2 group is refused.
        f2 = joinpath(dir, "f2.h5")
        h5open(f2, "w") do file
            spectrum_group = create_group(file, "spectrum")
            spectrum_group["quart31/index"] = reshape([1, 1, 1, 2], 4, 1)
            spectrum_group["metadata/schema_version"] = "cyaxiverse-spectrum-groups-1"
        end
        @test_throws ArgumentError _quartic_examine(f2, CYAxiverse.structs.GeometryIndex(2, 1, 1))
        h5open(file -> HDF5.delete_object(file["spectrum"], "metadata"), f2, "r+")
        records, _ = _quartic_examine(f2, CYAxiverse.structs.GeometryIndex(2, 1, 1))
        @test !isempty(records)
    end
end
