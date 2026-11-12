# CYAX-0232: the c = 1 supergravity axion kinetic normalization.
# R-003 (named source reproduction), R-004 (factor contract and orientation)
# and the R-005 c = 1/2 comparison for the routes these fixtures exercise.
# Fixture inputs are pinned in specs/0232-kinetic-normalization/plan.md.

using LinearAlgebra

# Eq. 3.18 instanton scales, as `leading_axion_reference_data` builds them
# (fuzzy_axions_model_stage.jl): log10[τ e^{-2πτ}] plus the constant
# prefactor log10(8π sqrt(P) m_{3/2} / V), with m_{3/2} in Planck units.
function _cyax0232_eq318_log10_lambda4(tau, cy_volume, gs, w0)
    PB = CYAxiverse.paper_benchmarks
    prefactor = PB.fuzzy_axion_prefactor_P(gs)
    gravitino = PB.fuzzy_axion_gravitino_mass(prefactor,
        PB.fuzzy_axion_kahler_potential(cy_volume), w0; mplanck_ev=1.0)
    L = PB.instanton_scales(Float64.(tau), 1.0)
    L[2, :] .+= log10(8π) + 0.5 * log10(Float64(prefactor)) +
        log10(Float64(gravitino)) - log10(Float64(cy_volume))
    (; L, prefactor, gravitino)
end

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
