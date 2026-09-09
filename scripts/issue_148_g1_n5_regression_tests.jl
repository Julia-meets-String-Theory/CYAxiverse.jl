using CYAxiverse
using Test
using Printf

poly102 = CYAxiverse.paper_benchmarks.poly102_inflation
n5_kc = poly102.n5_critical_scale()

n5_ratio_formula(k::Real) =
    (32 / (255 / 8)) * exp(-2π * k * (32 - 255 / 8))

@testset "N5 reduced zero-phase regression checks" begin
    n5_kc_path = poly102.n5_reduced_zero_phase_continuation(
        [n5_kc - 1e-3, n5_kc - 2e-4, n5_kc + 2e-4, n5_kc + 1e-3];
        seed_theta=π + 1.0e-3, max_iterations=64, gradient_tolerance=1e-10,
        hessian_tolerance=1e-10)

    @test isapprox(poly102.n5_reduced_ratio(n5_kc), 0.25; atol=1e-15)
    @test all(isapprox(step.ratio, n5_ratio_formula(step.k); rtol=1e-15, atol=1e-18)
        for step in n5_kc_path)
    @test all(step.branch == :pi for step in n5_kc_path)
    @test all(step.converged for step in n5_kc_path)
    @test all(step.gradient <= 1e-9 for step in n5_kc_path)

    catastrophe_index = findfirst(step -> step.catastrophe_detected, n5_kc_path)
    @test catastrophe_index !== nothing
    @test isapprox(n5_kc_path[catastrophe_index].catastrophe_k, n5_kc; atol=5e-7)
    @test abs(n5_kc_path[catastrophe_index].catastrophe_hessian) <= 1e-10
    @test n5_kc_path[catastrophe_index].catastrophe_residual <= 1e-10
    @test isapprox(n5_kc_path[catastrophe_index].catastrophe_theta, π; atol=1e-10)
    @test catastrophe_index > 1

    n5_tight_event_path = poly102.n5_reduced_zero_phase_continuation(
        [n5_kc - 1e-3, n5_kc - 2e-4, n5_kc + 2e-4, n5_kc + 1e-3];
        seed_theta=π + 1e-3, max_iterations=1,
        gradient_tolerance=1e-12, hessian_tolerance=1e-10,
        event_scale_tolerance=1e-14)
    @test !any(step -> step.catastrophe_detected, n5_tight_event_path)

    n5_nonconverged_path = poly102.n5_reduced_zero_phase_continuation(
        [0.1, 1.0]; seed_theta=π + 1e-3,
        max_iterations=1, gradient_tolerance=1e-16, hessian_tolerance=1e-10,
        event_scale_tolerance=1e-14)
    @test !n5_nonconverged_path[1].converged
    @test n5_nonconverged_path[1].iterations == 1
    @test !n5_nonconverged_path[1].catastrophe_detected

    n5_zero_seed_path = poly102.n5_reduced_zero_phase_continuation(
        [n5_kc - 1e-3, n5_kc + 1e-3]; seed_theta=1e-3)
    @test all(step.branch == :zero for step in n5_zero_seed_path)
    @test all(!step.catastrophe_detected for step in n5_zero_seed_path)

    @test_throws ArgumentError poly102.n5_reduced_zero_phase_continuation(
        [n5_kc - 1e-3, n5_kc + 1e-3]; seed_theta=3.101964568393247)

    @test poly102.n5_reduced_zero_phase_continuation([1e-3, 2e-3];
        seed_theta=1.0e-3, hessian_tolerance=1e-8)[1].branch == :zero

    n5_big_kc_path = poly102.n5_reduced_zero_phase_continuation(
        BigFloat.([n5_kc - 1e-3, n5_kc + 1e-3]); seed_theta=big(π) + 1e-3)
    @test typeof(first(n5_big_kc_path).k) == BigFloat

    @test_throws ArgumentError poly102.n5_reduced_zero_phase_continuation(
        [1.0, 2.0]; gradient_tolerance=-1.0)
    @test_throws ArgumentError poly102.n5_reduced_zero_phase_continuation(
        [1.0, 2.0]; hessian_tolerance=0.0)
    @test_throws ArgumentError poly102.n5_reduced_zero_phase_continuation(
        [1.0, 2.0]; event_scale_tolerance=0.0)
    @test_throws ArgumentError poly102.n5_reduced_zero_phase_continuation(
        [1.0, 2.0]; event_scale_tolerance=1 / 0)
end

@printf("n5_kc=%.20f\n", n5_kc)
@printf("n5_ratio_prefactor=%.20f\n", Float64(poly102._n5_reduced_zero_phase_ratio_prefactor(Float64)))
