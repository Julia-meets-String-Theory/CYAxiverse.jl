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
