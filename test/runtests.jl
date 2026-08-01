using CYAxiverse
using LinearAlgebra
using Test

@testset "CYAxiverse.jl" begin
    @testset "core" begin
        @test CYAxiverse.greet_CYAxiverse() == "Hello CYAxiverse!"
        @test CYAxiverse.greet_CYAxiverse() != "Hello world!"
    end

    if get(ENV, "CYAXIVERSE_TEST_CYTOOLS", "0") == "1"
        @testset "cytools wrapper regression repro" begin
            include(joinpath(@__DIR__, "..", "scripts", "cytools_wrapper_repro.jl"))
        end
    end
end

@testset "HP spectrum: one-axion analytic mass" begin
    # V = 10^-20 * (1 - cos(3θ)); second instanton is exactly absent.
    K = Hermitian(reshape([4.0], 1, 1))
    Q = reshape(Int[3, 6], 1, 2)
    L = [1.0 0.0;
         -20.0 -1000.0]

    # hp_spectrum's low-level interface uses one instanton per row.
    hp = CYAxiverse.generate.hp_spectrum(K, Matrix(L'), Matrix(Q'); prec=200)

    # λ = Λ q² / K, followed by hp_spectrum's output-unit conversion.
    expected_mass = 0.5 * (-20.0 + log10(3^2 / 4.0)) +
                    log10(2.435e18) + 9.0 + log10(2π)

    @test isapprox(hp["m"][1], expected_mass; atol=1e-10)
    @test hp["msign"][1] == 1

    # The canonical charge is q/sqrt(K) = 3/2, so the fourth derivative is
    # Lambda * (3/2)^4 in hp_spectrum's reported units.
    expected_quartic = -20.0 + 4 * log10(3 / 2) + 4 * log10(2π)
    @test hp["λselfsign"][1] == 1
    @test isapprox(hp["λself"][1], expected_quartic; atol=1e-10)
end
@testset "PQ and HP spectrum: diagonal two-axion comparison" begin
    # The leading instantons act on separate axions, so the PQ construction is
    # exact. The third, zero-sign instanton only satisfies N > h11 and makes
    # no contribution to the potential.
    K = Hermitian([4.0 0.0;
                   0.0 9.0])
    Q = [3 0 0;
         0 5 0]
    L = [1.0 1.0 0.0;
         -20.0 -30.0 -1000.0]

    pq = CYAxiverse.generate.pq_spectrum(K, L, Q)
    hp = CYAxiverse.generate.hp_spectrum(K, Matrix(L'), Matrix(Q'); prec=200)

    expected = sort([
        0.5 * (-20.0 + log10(3^2 / 4.0)),
        0.5 * (-30.0 + log10(5^2 / 9.0)),
    ] .+ log10(2.435e18) .+ 9.0 .+ log10(2π))

    @test all(isapprox.(pq.m, expected; atol=1e-10))
    @test all(isapprox.(hp["m"], expected; atol=1e-10))
    @test all(isapprox.(pq.m, hp["m"]; atol=1e-10))
    @test pq.λselfsign == hp["λselfsign"]
    @test all(isapprox.(pq.λself, hp["λself"]; atol=1e-10))
end

@testset "PQ spectrum: non-diagonal kinetic matrix" begin
    # A non-diagonal K catches the side on which the Cholesky inverse acts.
    K = Hermitian([4.0 1.2;
                   1.2 9.0])
    Q = [3 0 0;
         0 5 0]
    L = [1.0 1.0 0.0;
         -20.0 -30.0 -1000.0]

    pq = CYAxiverse.generate.pq_spectrum(K, L, Q)
    hp = CYAxiverse.generate.hp_spectrum(K, Matrix(L'), Matrix(Q'); prec=200)

    @test all(isapprox.(pq.m, hp["m"]; atol=0.1))
end

@testset "PQ-seeded HP alignment" begin
    K = Hermitian([4.0 0.0;
                   0.0 9.0])
    Q = [3 0 0;
         0 5 0]
    L = [1.0 1.0 0.0;
         -20.0 -30.0 -1000.0]

    alignment = CYAxiverse.generate.pq_hp_alignment(K, L, Q; prec=200)

    @test alignment.permutation == [1, 2]
    @test all(isapprox.(alignment.aligned_overlap, ones(2); atol=1e-12))
    @test all(alignment.residuals .< 1e-100)
end
