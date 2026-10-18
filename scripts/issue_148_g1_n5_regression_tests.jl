using CYAxiverse
using Test
using Printf

poly102 = CYAxiverse.paper_benchmarks.poly102_inflation
n5_kc = poly102.n5_critical_scale()

n5_ratio_formula(k::Real) =
    (32 / (255 / 8)) * exp(-2π * k * (32 - 255 / 8))

n5_closed_form_kc(T) = convert(T, 4) / convert(T, π) * log(convert(T, 1024) / convert(T, 255))

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

    n5_near_event_nonconvergence = poly102.n5_reduced_zero_phase_continuation(
        [n5_kc - 1e-3, n5_kc]; seed_theta=π + 1e-3, max_iterations=1,
        gradient_tolerance=1e-16, hessian_tolerance=1e-10, event_scale_tolerance=1e-14)
    @test !n5_near_event_nonconvergence[1].converged
    @test !n5_near_event_nonconvergence[1].catastrophe_detected
    @test !n5_near_event_nonconvergence[2].catastrophe_detected

    n5_endpoint_scale_check = poly102.n5_reduced_zero_phase_continuation(
        [n5_kc, n5_kc + 1e-11]; seed_theta=π + 1e-3,
        gradient_tolerance=1e-10, hessian_tolerance=1e-10,
        event_scale_tolerance=1e-14, max_iterations=2)
    @test !any(step -> step.catastrophe_detected, n5_endpoint_scale_check)

    n5_zero_seed_path = poly102.n5_reduced_zero_phase_continuation(
        [n5_kc - 1e-3, n5_kc + 1e-3]; seed_theta=1e-3)
    @test all(step.branch == :zero for step in n5_zero_seed_path)
    @test all(!step.catastrophe_detected for step in n5_zero_seed_path)

    @test_throws ArgumentError poly102.n5_reduced_zero_phase_continuation(
        [n5_kc - 1e-5, n5_kc - 1e-4]; seed_theta=3.101964568393247)

    k_near_cusp = n5_kc - 1e-5
    a_near_cusp = poly102.n5_reduced_ratio(k_near_cusp)
    sat_theta_near_cusp = acos(-1 / (4 * a_near_cusp))
    @test abs(sat_theta_near_cusp - π) < 0.01
    @test_throws ArgumentError poly102.n5_reduced_zero_phase_continuation(
        [k_near_cusp]; seed_theta=sat_theta_near_cusp)

    pi_near_cusp_path = poly102.n5_reduced_zero_phase_continuation(
        [k_near_cusp]; seed_theta=π + 1e-3)
    @test pi_near_cusp_path[1].branch == :pi
    @test pi_near_cusp_path[1].converged
    @test abs(pi_near_cusp_path[1].theta - π) < abs(sat_theta_near_cusp - π) / 2

    @test poly102.n5_reduced_zero_phase_continuation([1e-3, 2e-3];
        seed_theta=1.0e-3, hessian_tolerance=1e-8)[1].branch == :zero

    for bit_precision in (128, 256)
        setprecision(BigFloat, bit_precision) do
            local_kc = n5_closed_form_kc(BigFloat)
            local_path = poly102.n5_reduced_zero_phase_continuation(
                [local_kc - BigFloat("1.0e-3"), local_kc + BigFloat("1.0e-3")];
                seed_theta=big(π) + BigFloat("1e-3"),
                gradient_tolerance=BigFloat("1e-10"),
                hessian_tolerance=BigFloat("1e-10"),
                event_scale_tolerance=BigFloat("1e-10"))
            local_idx = findfirst(step -> step.catastrophe_detected, local_path)
            @test local_idx !== nothing
            @test isapprox(local_path[local_idx].catastrophe_k, local_kc; rtol=1e-10)
            @test abs(local_path[local_idx].catastrophe_hessian) <= BigFloat("1e-10")
            @test local_path[local_idx].catastrophe_residual <= BigFloat("1e-10")
            @test local_path[2].k isa BigFloat
            @test typeof(first(local_path).theta) == BigFloat
        end
    end

    setprecision(BigFloat, 256) do
        local_kc = n5_closed_form_kc(BigFloat)
        k_hp = local_kc - BigFloat("1e-20")
        cp_default = poly102.n5_reduced_critical_points(k_hp)
        @test length(cp_default.theta) == 4
        @test cp_default.minima == 2
        a_hp = poly102.n5_reduced_ratio(k_hp)
        sat_hp = acos(BigFloat(-1) / (4 * a_hp))
        @test abs(sat_hp - BigFloat(π)) > 0
        @test abs(sat_hp - BigFloat(π)) < BigFloat("1e-5")
    end

    setprecision(BigFloat, 256) do
        local_kc = n5_closed_form_kc(BigFloat)
        k_close = local_kc - BigFloat("1e-40")
        a_close = poly102.n5_reduced_ratio(k_close)
        cp_close = poly102.n5_reduced_critical_points(k_close; atol=zero(BigFloat))
        @test length(cp_close.theta) == 4
        @test cp_close.minima == 2
        lower_analytic = acos(BigFloat(-1) / (BigFloat(4) * a_close))
        upper_analytic = BigFloat(2) * BigFloat(π) - lower_analytic
        lower_offset = lower_analytic - BigFloat(π)
        upper_offset = upper_analytic - BigFloat(π)
        @test isapprox(lower_offset, -upper_offset; rtol=BigFloat("1e-60"))
        @test any(isapprox(t, lower_analytic; atol=BigFloat("1e-70")) for t in cp_close.theta)
        @test any(isapprox(t, upper_analytic; atol=BigFloat("1e-70")) for t in cp_close.theta)
        grad(t) = sin(t) + BigFloat(2) * a_close * sin(BigFloat(2) * t)
        for t in cp_close.theta
            @test abs(grad(t)) < BigFloat("1e-70")
        end
    end

    r_rational = poly102.n5_reduced_ratio(177 // 100)
    @test r_rational isa BigFloat
    @test isapprox(Float64(r_rational), poly102.n5_reduced_ratio(1.77); atol=1e-14)
    e_rational = poly102.n5_reduced_exponent(177 // 100)
    @test e_rational isa BigFloat
    cp_rational = poly102.n5_reduced_critical_points(177 // 100)
    @test cp_rational.theta[1] isa BigFloat
    r_ratbig = poly102.n5_reduced_ratio(big(177) // big(100))
    @test r_ratbig isa BigFloat

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
