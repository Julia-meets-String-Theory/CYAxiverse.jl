#!/usr/bin/env julia

using LinearAlgebra
using Printf

using CYAxiverse

poly102 = CYAxiverse.paper_benchmarks.poly102_inflation

@inline n5_closed_form_ratio(k) = begin
    ratio = (BigFloat(32) / (BigFloat(255) / BigFloat(8))) *
        exp(-BigFloat(2) * BigFloat(π) * k * (BigFloat(32) - BigFloat(255) / BigFloat(8)))
    ratio
end

@inline n5_closed_form_kc() = begin
    convert(BigFloat, 4) / BigFloat(π) * log(BigFloat(1024) / BigFloat(255))
end

function capture(predicate::Function)
    try
        result = predicate()
        return result, nothing
    catch err
        return nothing, err
    end
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
    @printf("catastrophe_scale_error=%.3e\n", Float64(abs(step.catastrophe_k - n5_kc)))
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
    @printf("%s_step_%d_catastrophe_residual=%.3e\n", prefix, index, Float64(step.catastrophe_residual))
    @printf("%s_step_%d_catastrophe_hessian=%.3e\n", prefix, index, Float64(step.catastrophe_hessian))
    @printf("%s_step_%d_catastrophe_scale_error=%.3e\n", prefix, index, Float64(abs(step.catastrophe_k - n5_kc)))
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

println("secondary_branch_len=$(length(poly102.n5_reduced_zero_phase_continuation(
    [n5_kc - 1e-3, n5_kc + 1e-3]; seed_theta=1.0e-3)))")

# near-cusp satellite rejection must fail as unsupported branch support
near_satellite_rejected, near_satellite_err = capture() do
    poly102.n5_reduced_zero_phase_continuation(
        [n5_kc - 1e-5, n5_kc + 1e-5]; seed_theta=3.101964568393247)
    :not_raised
end
println("invalid_near_satellite_seed_rejected=$(near_satellite_rejected === nothing)")
if near_satellite_err !== nothing
    println("invalid_near_satellite_err=$(typeof(near_satellite_err))")
end

# far satellite seed remains outside π-window and must still be rejected by seed validator
far_satellite_rejected, far_satellite_err = capture() do
    poly102.n5_reduced_zero_phase_continuation(
        [n5_kc - 1e-3, n5_kc + 1e-3]; seed_theta=3.101964568393247)
    :not_raised
end
println("invalid_farsatellite_seed_rejected=$(far_satellite_rejected === nothing)")
if far_satellite_err !== nothing
    println("invalid_farsatellite_err=$(typeof(far_satellite_err))")
end

# Nonconverged near-event boundary: should not propagate catastrophe
near_event_nonconv, near_event_err = capture() do
    path = poly102.n5_reduced_zero_phase_continuation(
        [n5_kc - 1e-3, n5_kc]; seed_theta=π + 1e-3,
        max_iterations=1, gradient_tolerance=1e-16,
        hessian_tolerance=1e-10, event_scale_tolerance=1e-14)
    print_step_summary("near_event_nonconv", path[1], 1)
    print_step_summary("near_event_nonconv", path[2], 2)
    @printf("near_event_nonconv_catastrophe=%s\n", path[2].catastrophe_detected)
    :not_raised
end
if near_event_err !== nothing
    println("near_event_nonconv_err=$near_event_err")
end

# Endpoint scale contract check
endpoint_scale_path, endpoint_scale_err = capture() do
    poly102.n5_reduced_zero_phase_continuation(
        [n5_kc, n5_kc + 1e-11]; seed_theta=π + 1e-3,
        gradient_tolerance=1e-10, hessian_tolerance=1e-10,
        event_scale_tolerance=1e-14)
end
println("endpoint_scale_catastrophe_detected=$(findfirst(step -> step.catastrophe_detected, endpoint_scale_path) !== nothing)")
println("endpoint_scale_path_second_event=" * (endpoint_scale_path === nothing ? "none" : string(endpoint_scale_path[2].catastrophe_detected)))

@printf("mixed_type_path_eltype=%s\n",
    string(typeof(first(poly102.n5_reduced_zero_phase_continuation(
        BigFloat.([n5_kc - 1e-3, n5_kc + 1e-3]); seed_theta=big(π) + 1e-3)).theta)))

critical_below = poly102.n5_reduced_critical_points(n5_kc - 1e-3)
critical_above = poly102.n5_reduced_critical_points(n5_kc + 1e-3)
println("critical_minima_below=$(critical_below.minima)")
println("critical_minima_above=$(critical_above.minima)")

# Branch identity checks
pi_path = poly102.n5_reduced_zero_phase_continuation(
    [n5_kc - 1e-3, n5_kc + 1e-3]; seed_theta=π + 1e-3)
println("pi_seed_branch_check=$((pi_path[1].branch === :pi && pi_path[2].branch === :pi))")
zero_path = poly102.n5_reduced_zero_phase_continuation(
    [n5_kc - 1e-3, n5_kc + 1e-3]; seed_theta=1.0e-3)
println("zero_seed_branch_check=$((zero_path[1].branch === :zero && zero_path[2].branch === :zero))")

# Precision ladder replay with target-precision critical scale and closed-form orbit
for bits in (0, 128, 256)
    if bits == 0
        local_kc = n5_kc
        kgrid_local = [local_kc - 1e-3, local_kc - 2e-4, local_kc + 2e-4, local_kc + 1e-3]
        path_local = poly102.n5_reduced_zero_phase_continuation(
            kgrid_local; seed_theta=π + 1e-3,
            gradient_tolerance=1e-10, hessian_tolerance=1e-10, event_scale_tolerance=1e-10)
    else
        setprecision(BigFloat, bits) do
            local_kc = n5_closed_form_kc()
            kgrid_local = [local_kc - BigFloat("1.0e-3"), local_kc - BigFloat("2.0e-4"),
                local_kc + BigFloat("2.0e-4"), local_kc + BigFloat("1.0e-3")]
            path_local = poly102.n5_reduced_zero_phase_continuation(
                kgrid_local; seed_theta=big(π) + BigFloat("1e-3"),
                gradient_tolerance=BigFloat("1e-10"), hessian_tolerance=BigFloat("1e-10"),
                event_scale_tolerance=BigFloat("1e-10"))
        end
    end
    idx = findfirst(step -> step.catastrophe_detected, path_local)
    if idx === nothing
        println("precision_ladder_$(bits)_catastrophe_detected=false")
    else
        step = path_local[idx]
        @printf("precision_ladder_%d_catastrophe=%d\n", bits, idx)
        @printf("precision_ladder_%d_event_scale_error=%.3e\n", bits, Float64(abs(step.catastrophe_k - n5_closed_form_kc())))
        @printf("precision_ladder_%d_event_gradient=%.3e\n", bits, Float64(step.catastrophe_residual))
        @printf("precision_ladder_%d_event_hessian=%.3e\n", bits, Float64(step.catastrophe_hessian))
    end
end

println("validation_checks_complete=true")
