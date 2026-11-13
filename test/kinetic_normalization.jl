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
    # No route fails except the A1 cubic assertion of hp_spectrum_save on the
    # R-003 charges (N > h11), which T030 fixes; it fails the same way in both
    # conventions.
    errors = sort([k for k in keys(R005_C1) if endswith(k, "/error")])
    @test errors == ["r003/F2_hp/error"]
    @test r005_same("r003/F2_hp/error")
    @test Set(keys(R005_C1)) == Set(keys(R005_CHALF))
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
        fixture == "r003" && continue
        @test r005_offset("$fixture/F2_hp/masses/log10", -R005_H)
        @test r005_offset("$fixture/F2_hp/decay/fK", R005_H)
        @test r005_offset("$fixture/F2_hp/decay/fpert", R005_H)
        for group in ("quartdiag", "quart31", "quart22")
            @test r005_offset("$fixture/F2_hp/$group/log10", -4R005_H)
            @test r005_same("$fixture/F2_hp/$group/sign")
        end
        # The hp_spectrum_save cubic is in the ϑ/charge basis: factor 1.
        @test r005_same("$fixture/F2_hp/cubic/tensor")
        # The pq_spectrum_save cubic (factor 2^{-3/2}) is asserted by T030 on
        # the A1.4 cubic fixtures, after the A1 orientation fix.
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
