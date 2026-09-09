#!/usr/bin/env julia

using LinearAlgebra
using Printf

using CYAxiverse

poly102 = CYAxiverse.paper_benchmarks.poly102_inflation

@inline n5_closed_form_ratio(k) = begin
    ratio = (BigFloat(32) / (BigFloat(255) / BigFloat(8))) *
        exp(-BigFloat(2) * BigFloat(pi) * BigFloat(k) *
            (BigFloat(32) - BigFloat(255) / BigFloat(8)))
    ratio
end

function capture(predicate::Function)
    result = try
        predicate()
    catch err
        println(err)
        return false
    end
    return result
end

function print_event(path)
    idx = findfirst(step -> step.catastrophe_detected, path)
    if idx === nothing
        println("catastrophe_detected=false")
        return
    end
    step = path[idx]
    @printf("catastrophe_detected=true\n")
    @printf("catastrophe_index=%d\n", idx)
    @printf("catastrophe_k=%.20f\n", Float64(step.catastrophe_k))
    @printf("catastrophe_theta=%.18f\n", Float64(step.catastrophe_theta))
    @printf("catastrophe_residual=%.3e\n", Float64(step.catastrophe_residual))
    @printf("catastrophe_hessian=%.3e\n", Float64(step.catastrophe_hessian))
    @printf("catastrophe_scale_error=%.3e\n",
        Float64(abs(step.catastrophe_k - poly102.n5_critical_scale())))
    @printf("catastrophe_branch=%s\n", step.catastrophe_theta == π ? "pi" : string(step.branch))
end

function print_step_summary(prefix::AbstractString, step, index::Int)
    @printf("%s_step_%d_k=%.20f\n", prefix, index, Float64(step.k))
    @printf("%s_step_%d_ratio=%.20e\n", prefix, index, Float64(step.ratio))
    @printf("%s_step_%d_gradient=%.3e\n", prefix, index, Float64(step.gradient))
    @printf("%s_step_%d_hessian=%.3e\n", prefix, index, Float64(step.hessian))
    @printf("%s_step_%d_theta=%.18f\n", prefix, index, Float64(step.theta))
    @printf("%s_step_%d_branch=%s\n", prefix, index, string(step.branch))
    @printf("%s_step_%d_converged=%s\n", prefix, index, step.converged)
    @printf("%s_step_%d_iterations=%d\n", prefix, index, step.iterations)
    @printf("%s_step_%d_catastrophe_detected=%s\n", prefix, index, step.catastrophe_detected)
end

n5_kc = poly102.n5_critical_scale()
@printf("n5_kc=%.20f\n", n5_kc)
@printf("n5_ratio_at_kc=%.20e\n", Float64(poly102.n5_reduced_ratio(n5_kc)))

kgrid = [n5_kc - 1e-3, n5_kc - 2e-4, n5_kc + 2e-4, n5_kc + 1e-3]
path = poly102.n5_reduced_zero_phase_continuation(
    kgrid; seed_theta=π + 1e-3, gradient_tolerance=1e-10,
    hessian_tolerance=1e-10, max_iterations=64)

println("default_path_len=$(length(path))")
for (index, step) in enumerate(path)
    print_step_summary("default", step, index)
end
print_event(path)

@printf("ratio_prefactor=%.20f\n", Float64(poly102._n5_reduced_zero_phase_ratio_prefactor(Float64)))
@printf("ratio_formula_at_kc_minus_1e-3=%.20e\n",
    Float64(poly102.n5_reduced_ratio(n5_kc - 1e-3)))
@printf("ratio_formula_at_kc_plus_1e-3=%.20e\n",
    Float64(poly102.n5_reduced_ratio(n5_kc + 1e-3)))

setprecision(BigFloat, 256) do
    kc_big = BigFloat(n5_kc)
    k_test = kc_big - BigFloat("1.0e-3")
    ratio_lib = poly102.n5_reduced_ratio(k_test)
    ratio_ref = n5_closed_form_ratio(k_test)
    diff = abs(ratio_lib - ratio_ref)
    @printf("n5_ratio_mixed_max_error_256=%.5e\n", Float64(diff))
    @printf("n5_ratio_mixed_k=%.30e\n", Float64(ratio_lib))
    @printf("n5_ratio_closed_form_mixed_k=%.30e\n", Float64(ratio_ref))
end

path_tight = poly102.n5_reduced_zero_phase_continuation(
    kgrid; seed_theta=π + 1e-3, max_iterations=1,
    gradient_tolerance=1e-12, hessian_tolerance=1e-10,
    event_scale_tolerance=1e-14)
println("default_path_tight_event_detected=$(findfirst(step -> step.catastrophe_detected, path_tight) !== nothing)")

n5_nonconverged = poly102.n5_reduced_zero_phase_continuation(
    [0.1, 1.0]; seed_theta=π + 1e-3,
    max_iterations=1, gradient_tolerance=1e-16,
    hessian_tolerance=1e-10, event_scale_tolerance=1e-14)
println("nonconverged_first_step_converged=$(n5_nonconverged[1].converged)")
println("nonconverged_first_step_iterations=$(n5_nonconverged[1].iterations)")
println("nonconverged_second_step_converged=$(n5_nonconverged[2].converged)")
println("nonconverged_first_step_catastrophe=$(n5_nonconverged[1].catastrophe_detected)")

@printf("secondary_branch_len=%d\n",
    length(poly102.n5_reduced_zero_phase_continuation(
        [n5_kc - 1e-3, n5_kc + 1e-3]; seed_theta=1.0e-3)))

invalid_seed_ok = capture() do
    poly102.n5_reduced_zero_phase_continuation(
        [n5_kc - 1e-3, n5_kc + 1e-3]; seed_theta=3.101964568393247)
    false
end
println("invalid_satellite_seed_rejected=$(!invalid_seed_ok)")

println("mixed_type_path_eltype=$(typeof(first(poly102.n5_reduced_zero_phase_continuation(
    BigFloat.([n5_kc - 1e-3, n5_kc + 1e-3]); seed_theta=big(π) + 1e-3)).k))")

critical_below = poly102.n5_reduced_critical_points(n5_kc - 1e-3)
critical_above = poly102.n5_reduced_critical_points(n5_kc + 1e-3)
println("critical_minima_below=$(critical_below.minima)")
println("critical_minima_above=$(critical_above.minima)")

# Branch identity check
pi_path = poly102.n5_reduced_zero_phase_continuation(
    [n5_kc - 1e-3, n5_kc + 1e-3]; seed_theta=π + 1e-3)
println("pi_seed_branch_check=$((pi_path[1].branch === :pi && pi_path[2].branch === :pi))")
zero_path = poly102.n5_reduced_zero_phase_continuation(
    [n5_kc - 1e-3, n5_kc + 1e-3]; seed_theta=1.0e-3)
println("zero_seed_branch_check=$((zero_path[1].branch === :zero && zero_path[2].branch === :zero))")

println("validation_checks_complete=true")
