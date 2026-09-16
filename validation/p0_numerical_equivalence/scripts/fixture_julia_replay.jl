#!/usr/bin/env julia
"""Replay the P0 fixture corpus through the pinned Julia source routes.

This is an evidence harness, not a replacement numerical kernel.  It uses the
retained normalized P0 environment to load CYAxiverse and calls historical
package functions where the route is callable.  Source-level probes reproduce
the exact local formulas where a result is not exposed by the public return
value (for example row scales, H3, and direct `-Inf` classification).
"""

using TOML
using LinearAlgebra

using CYAxiverse

const SOURCE_REVISION = "7a40285bb5c313f7e8746b90644d5f45bb67be44"
const ROOT = normpath(joinpath(@__DIR__, ".."))
const FIXTURE_DIR = joinpath(ROOT, "fixtures")
const MIN_LOGSCALE = log10(floatmin(Float64))

compact(value) = replace(sprint(show, value), '\n' => ' ')

function as_float(value)
    value isa String || return Float64(value)
    value == "-Inf" && return -Inf
    value == "+Inf" && return Inf
    parse(Float64, value)
end

function matrix_of(rows, ::Type{T}) where {T}
    isempty(rows) && return Matrix{T}(undef, 0, 0)
    result = Matrix{T}(undef, length(rows), length(first(rows)))
    for (row_index, row) in enumerate(rows), (column_index, value) in enumerate(row)
        result[row_index, column_index] = value isa String ? T(as_float(value)) : T(value)
    end
    result
end

function fixture_inputs(data)
    q = matrix_of(data["q"], Int)
    L = matrix_of(data["l"], Float64)
    phases = haskey(data, "phases") ? Float64.(as_float.(data["phases"])) : zeros(size(q, 2))
    theta = haskey(data, "theta") ? Float64.(as_float.(data["theta"])) : nothing
    basis = haskey(data, "coordinate_basis") ? matrix_of(data["coordinate_basis"], Float64) : nothing
    q_float = Float64.(q)
    transformed = basis === nothing ? q_float : basis \ q_float
    (; q, L, phases, theta, transformed)
end

function row_logscale(q, logs)
    [isempty(support) ? maximum(logs) : maximum(logs[support]) for support in
        ([j for j in axes(q, 2) if !iszero(q[i, j])] for i in axes(q, 1))]
end

function scaled_amplitudes(q, coefficients, logs, row_logs)
    result = zeros(Float64, size(q))
    for i in axes(q, 1), j in axes(q, 2)
        q[i, j] == 0 && continue
        delta = logs[j] - row_logs[i]
        delta >= MIN_LOGSCALE && (result[i, j] = coefficients[j] * 10.0^delta)
    end
    result
end

function global_amplitudes(coefficients, logs)
    shift = maximum(logs)
    coefficients .* 10.0 .^ (logs .- shift)
end

function h3_probe(q, coefficients, logs, row_logs, theta, phases)
    n, p = size(q)
    arguments = (2π .* (q' * theta)) .+ phases
    hessian = zeros(Float64, n, n)
    for row in 1:n, column in 1:n, term in 1:p
        (q[row, term] == 0 || q[column, term] == 0) && continue
        delta = logs[term] - (row_logs[row] + row_logs[column]) / 2
        delta < MIN_LOGSCALE && continue
        hessian[row, column] += q[row, term] * q[column, term] * coefficients[term] *
            10.0^delta * cos(arguments[term])
    end
    hessian .*= (2π)^2
    (; arguments, hessian)
end

function source_probe(data, inputs)
    q = inputs.transformed
    coefficients = vec(inputs.L[1, :])
    logs = vec(inputs.L[2, :])
    scales = row_logscale(q, logs)
    result = (; row_logscale=scales, support=map(!iszero, q),
        scaled_amplitudes=scaled_amplitudes(q, coefficients, logs, scales),
        global_amplitudes=global_amplitudes(coefficients, logs))
    inputs.theta === nothing && return result
    hessian = h3_probe(q, coefficients, logs, scales, inputs.theta, inputs.phases)
    merge(result, (; arguments=hessian.arguments, h3=hessian.hessian))
end

function show_direct_result(data, inputs)
    theta = inputs.theta
    seed = theta === nothing ? zeros(Float64, size(inputs.q, 1)) : theta
    try
        result = CYAxiverse.minimizer.critical_points(inputs.L, inputs.q;
            phases=inputs.phases, starts=1, max_iterations=40,
            initial_points=reshape(seed, :, 1))
        println("direct_critical_points=accepted",
            " roots=", result.critical_count,
            " minima=", result.minima_count,
            " inertia=", compact(result.inertia),
            " h3_eigenvalues=", compact(result.hessian_eigenvalues))
    catch error
        println("direct_critical_points=error ", sprint(showerror, error))
    end
end

function show_validator_results(inputs)
    try
        CYAxiverse.generate.logshifted_derivative_workspace(inputs.q, inputs.L)
        println("workspace=accepted")
    catch error
        println("workspace=error ", sprint(showerror, error))
    end
    K = Matrix{Float64}(I, size(inputs.q, 1), size(inputs.q, 1))
    try
        CYAxiverse.inflation_points.prepare_context(inputs.q, inputs.L, K)
        println("inflation_context=accepted")
    catch error
        println("inflation_context=error ", sprint(showerror, error))
    end
end

function show_f6_route(data, inputs)
    hilltop = Float64.(as_float.(data["hilltop"]))
    displacement = as_float(data["displacement"])
    sign = as_float(data["displacement_sign"])
    direction = Float64.(as_float.(data["mass_direction"]))
    expected = mod.(hilltop .+ sign * displacement .* direction, 1.0)
    println("source_displacement_theta_initial=", compact(expected))
    K = Matrix{Float64}(I, size(inputs.q, 1), size(inputs.q, 1))
    try
        context = CYAxiverse.inflation_points.prepare_context(inputs.q, inputs.L, K)
        actual_mass = CYAxiverse.inflation_points.mass_eigenbasis(context, hilltop; vectors=true)
        println("mass_eigenbasis=accepted eigenvalues=", compact(actual_mass.eigenvalues),
            " metric_residual=", actual_mass.metric_residual)
        supplied_mass = (; eigenvalues=[1.0, 2.0], raw_eigenvectors=Matrix{Float64}(I, 2, 2))
        flow = CYAxiverse.inflation_points.gradient_flow(context, hilltop;
            displacement, displacement_sign=sign, mass_basis=supplied_mass,
            mode_index=1, max_efolds=1e-3, step=1e-3)
        println("gradient_flow=", flow.status,
            " theta_initial=", compact(flow.theta_initial),
            " chart=", flow.coordinate_chart,
            " steps=", flow.steps)
    catch error
        println("inflation_route=error ", sprint(showerror, error))
    end
end

function show_f9_workspace(inputs)
    workspace = CYAxiverse.generate.logshifted_derivative_workspace(inputs.q, inputs.L)
    first = CYAxiverse.generate.logshifted_derivatives!(workspace, inputs.theta, inputs.q)
    retained_gradient = first.gradient
    retained_hessian = first.hessian
    retained_gradient_before = copy(retained_gradient)
    retained_hessian_before = copy(retained_hessian)
    second_theta = zeros(Float64, size(inputs.q, 1))
    second = CYAxiverse.generate.logshifted_derivatives!(workspace, second_theta, inputs.q)
    println("workspace_first=value=", first.value,
        " gradient_before=", compact(retained_gradient_before),
        " hessian_before=", compact(retained_hessian_before))
    println("workspace_aliasing=gradient_same=", retained_gradient === second.gradient,
        " hessian_same=", retained_hessian === second.hessian,
        " gradient_after=", compact(retained_gradient),
        " hessian_after=", compact(retained_hessian))
end

function source_f10_thresholds(probe)
    values = eigvals(Symmetric(probe.h3))
    seed_threshold = -100 * eps(Float64)
    seed_modes = count(<(seed_threshold), values)
    scale = max(maximum(abs, values), 1.0)
    final_tolerance = 100 * 1e-10 * scale
    final_inertia = (count(<(-final_tolerance), values),
        count(x -> abs(x) <= final_tolerance, values),
        count(>(final_tolerance), values))
    println("seed_threshold=", seed_threshold,
        " seed_negative_modes=", seed_modes,
        " final_zero_tolerance=", final_tolerance,
        " final_inertia=", final_inertia)
end

function show_f11_route(data)
    benchmark = CYAxiverse.paper_benchmarks.n5_potential(k=1.0)
    package_kc = CYAxiverse.paper_benchmarks.n5_critical_scale()
    legacy_kc = CYAxiverse.paper_benchmarks.author_inflation.n5_critical_scale()
    source_kc = 4 / π * log(1024 / 255)
    source_ratio(k) = (32 / (255 / 8)) * exp(-2π * k * (32 - 255 / 8))
    source_count(k) = begin
        a = source_ratio(k)
        points = Float64[0, π]
        a > 1 / 4 + 64eps(Float64) && append!(points, (acos(-1 / (4a)), 2π - acos(-1 / (4a))))
        curvature = cos.(points) .+ 4a .* cos.(2 .* points)
        count(>(0), curvature)
    end
    below = CYAxiverse.paper_benchmarks.n5_reduced_critical_points(package_kc - 1e-4)
    above = CYAxiverse.paper_benchmarks.n5_reduced_critical_points(package_kc + 1e-4)
    legacy_below = CYAxiverse.paper_benchmarks.author_inflation.n5_reduced_critical_points(legacy_kc - 1e-4)
    legacy_above = CYAxiverse.paper_benchmarks.author_inflation.n5_reduced_critical_points(legacy_kc + 1e-4)
    println("package_n5_potential=accepted Qsize=", size(benchmark.Q),
        " qdot_tau=", compact(benchmark.qdotτ),
        " package_kc=", package_kc,
        " package_minima_below=", below.minima,
        " package_minima_above=", above.minima)
    println("source_reduced_kc=", source_kc,
        " source_minima_below=", source_count(source_kc - 1e-4),
        " source_minima_above=", source_count(source_kc + 1e-4),
        " fixture_kc=", as_float(data["k_c"]))
    println("legacy_author_kc=", legacy_kc,
        " legacy_minima_below=", legacy_below.minima,
        " legacy_minima_above=", legacy_above.minima)
end

function main()
    current = strip(read(`git rev-parse HEAD`, String))
    source_equal = success(run(ignorestatus(`git diff --quiet $SOURCE_REVISION HEAD -- Project.toml src`)))
    println("julia_version=", VERSION)
    println("current_commit=", current)
    println("source_revision=", SOURCE_REVISION)
    println("source_tree_equal=", source_equal)
    println("package_load=ok")

    for path in sort(filter(name -> startswith(name, "F") && endswith(name, ".toml"), readdir(FIXTURE_DIR)))
        data = TOML.parsefile(joinpath(FIXTURE_DIR, path))
        id = data["id"]
        println("\n[", id, "]")
        if id == "F11-governed-n5-source-fixture"
            show_f11_route(data)
            continue
        end
        inputs = fixture_inputs(data)
        probe = source_probe(data, inputs)
        println("source_probe=row_logscale=", compact(probe.row_logscale),
            " support=", compact(probe.support),
            " scaled_amplitudes=", compact(probe.scaled_amplitudes),
            " global_amplitudes=", compact(probe.global_amplitudes))
        hasproperty(probe, :arguments) && println("source_probe=arguments=", compact(probe.arguments),
            " h3=", compact(probe.h3))
        show_direct_result(data, inputs)
        if id == "F4-exact-zero-and-minus-inf" || id == "F13-zero-only-support-all-minus-inf"
            show_validator_results(inputs)
        elseif id == "F6-coordinate-displacement"
            show_f6_route(data, inputs)
        elseif id == "F9-signed-cancellation"
            show_f9_workspace(inputs)
        elseif id == "F10-near-degenerate-classification"
            try
                source_f10_thresholds(probe)
            catch error
                println("threshold_probe=error ", sprint(showerror, error))
            end
        elseif id == "F7-transformed-near-zero-support" || id == "F8-transformed-noninteger-charge-map"
            println("transformed_q_exact=", compact(inputs.transformed))
        end
    end
end

main()
