using CYAxiverse
using Test

@testset "CYAxiverse.jl" begin
    @testset "CYAxiverse.jl" begin
        @test CYAxiverse.greet_CYAxiverse() == "Hello CYAxiverse!"
        @test CYAxiverse.greet_CYAxiverse() != "Hello world!"
    end

    @testset "cytools wrapper regression repro" begin
        include(joinpath(@__DIR__, "..", "scripts", "cytools_wrapper_repro.jl"))
    end
end

@testset "PQ and HP agree for one axion" begin
    K = Hermitian([4.0])
    Q = reshape([3], 1, 1)
    L = reshape([1.0, -20.0], 1, 2)

    pq = CYAxiverse.generate.pq_spectrum(K, L, Q)
    hp = CYAxiverse.generate.hp_spectrum(K, L, Q; prec=200)

    @test isapprox(pq.m[1], hp["m"][1]; atol=1e-10)
end
