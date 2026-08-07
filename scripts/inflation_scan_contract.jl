#!/usr/bin/env julia

"""Exercise the script-level inflation screening call contract.

This driver intentionally contains the generic orchestration. It calls the
existing package APIs but does not add a generic scan function to the package.
The current contract is:

1. `read.potential(GeometryIndex)` loads `L`, `Q`, and `K`.
2. `generate.LQtilde(Q, L)` selects the leading independent charges.
3. `generate.instanton_hierarchy_diagnostics(L)` supplies a cheap hierarchy
   diagnostic.
4. `generate.leading_hessian_mass_basis_float64(K, Ltilde, Qtilde)` supplies a
   cheap Float64 mass-basis diagnostic.
5. `generate.leading_critical_branches(selected; max_branches=...)` enumerates
   bounded leading branches for full-potential screening.

The local derivative/classification code below uses log-shifted amplitudes so
that screening ratios remain finite for hierarchically suppressed instantons.
It is deliberately separate from the package's physical trajectory solver;
the generic trajectory/refinement call is not locked down yet.

Usage:

```text
julia --project=. scripts/inflation_scan_contract.jl \
    --data-dir DATA_ROOT --geometry H,P,F [--geometry H,P,F ...]
```
"""

using CYAxiverse
using LinearAlgebra
using Printf
using Statistics

const GeometryIndex = CYAxiverse.structs.GeometryIndex

function _usage()
    println("Usage: julia --project=. scripts/inflation_scan_contract.jl " *
        "--data-dir DATA_ROOT --geometry H,P,F [--geometry H,P,F ...] " *
        "[--max-branches N]")
end

function _parse_args(args)
    data_dir = get(ENV, "CYAXIVERSE_DATA_DIR", "")
    geometries = GeometryIndex[]
    max_branches = 1_000_000
    index = 1
    while index <= length(args)
        arg = args[index]
        if arg in ("--help", "-h")
            _usage()
            exit(0)
        elseif arg in ("--data-dir", "--geometry", "--max-branches")
            index == length(args) && error("missing value for $arg")
            value = args[index + 1]
            if arg == "--data-dir"
                data_dir = value
            elseif arg == "--geometry"
                parts = parse.(Int, split(value, ','))
                length(parts) == 3 || error("--geometry must be H,P,F")
                push!(geometries, GeometryIndex(parts...))
            else
                max_branches = parse(Int, value)
            end
            index += 2
        else
            error("unknown argument $arg")
        end
    end
    isempty(data_dir) && error("--data-dir or CYAXIVERSE_DATA_DIR is required")
    isempty(geometries) && error("at least one --geometry H,P,F is required")
    max_branches > 0 || error("--max-branches must be positive")
    (; data_dir=abspath(data_dir), geometries, max_branches)
end

function _timed_call(f)
    GC.gc(false)
    measured = @timed f()
    (; value=measured.value, seconds=measured.time, bytes=measured.bytes)
end

function _oriented_potential(geom_idx::GeometryIndex)
    potential = CYAxiverse.read.potential(geom_idx)
    Q = Matrix{Int}(potential.Q)
    L = Matrix{Float64}(potential.L)
    if size(L, 1) != 2 && size(L, 2) == 2
        L = Matrix(L')
    end
    if size(Q, 2) != size(L, 2) && size(Q, 1) == size(L, 2)
        Q = Matrix(Q')
    end
    size(L, 1) == 2 || throw(DimensionMismatch("L must have two rows"))
    size(Q, 2) == size(L, 2) ||
        throw(DimensionMismatch("Q and L must have the same instanton count"))
    size(Q, 1) == size(potential.K, 1) ||
        throw(DimensionMismatch("Q and K must have the same axion count"))
    Q, L, Hermitian(Matrix{Float64}(potential.K))
end

function _normalized_derivatives(theta::AbstractVector{<:Real},
        Q::Matrix{Int}, L::Matrix{Float64})
    logs = @view L[2, :]
    shift = maximum(logs)
    amplitudes = L[1, :] .* 10.0 .^ (logs .- shift)
    arguments = 2π .* (Q' * theta)
    weighted_sines = amplitudes .* sin.(arguments)
    weighted_cosines = amplitudes .* cos.(arguments)
    (; value=sum(amplitudes .* (1 .- cos.(arguments))),
       gradient=2π .* Q * weighted_sines,
       hessian=(2π)^2 .* Q * Diagonal(weighted_cosines) * Q',
       amplitudes, arguments, log_shift=shift)
end

function _classify_point(theta, Q, L, Kfactor)
    derivatives = _normalized_derivatives(theta, Q, L)
    canonical_hessian = Kfactor.L \ derivatives.hessian / Kfactor.L'
    eigenvalues = eigvals(Symmetric(canonical_hessian))
    inverse_metric_gradient = Kfactor.L' \ (Kfactor.L \ derivatives.gradient)
    gradient_norm = sqrt(max(dot(derivatives.gradient, inverse_metric_gradient), 0.0))
    value = derivatives.value
    epsilon = value == 0 ? Inf : 0.5 * (gradient_norm / abs(value))^2
    eta_values = value == 0 ? fill(Inf, length(eigenvalues)) : eigenvalues ./ value
    (; value, gradient_norm, epsilon,
       min_eta=minimum(eta_values), max_eta=maximum(eta_values),
       abs_min_eta=minimum(abs.(eta_values)),
       negative_modes=count(<(0), eigenvalues),
       zeroish_modes=count(x -> abs(x) <=
           1e-10 * max(maximum(abs, eigenvalues), 1.0), eigenvalues),
       positive_modes=count(>(0), eigenvalues))
end

function _classify_branches(branches, Q, L, Kfactor)
    saddle_count = 0
    candidate_count = 0
    least_tachyonic = nothing
    best = nothing
    flattest = nothing
    for index in axes(branches.coordinates, 2)
        classification = _classify_point(
            @view(branches.coordinates[:, index]), Q, L, Kfactor)
        classification.negative_modes > 0 || continue
        classification.value > 0 || continue
        saddle_count += 1
        classification.epsilon < 1 && abs(classification.min_eta) < 1 &&
            (candidate_count += 1)
        least_tachyonic = least_tachyonic === nothing ||
            classification.min_eta > least_tachyonic.min_eta ? classification : least_tachyonic
        best = best === nothing || abs(classification.min_eta + 1) < abs(best.min_eta + 1) ?
            classification : best
        flattest = flattest === nothing || classification.abs_min_eta < flattest.abs_min_eta ?
            classification : flattest
    end
    (; saddle_count, candidate_count, least_tachyonic, best, flattest)
end

function run_geometry(geom_idx::GeometryIndex; max_branches::Int=1_000_000)
    started = time_ns()
    loaded = _timed_call(() -> _oriented_potential(geom_idx))
    Q, L, K = loaded.value

    selected = _timed_call(() -> CYAxiverse.generate.LQtilde(Q, L))
    hierarchy = _timed_call(() ->
        CYAxiverse.generate.instanton_hierarchy_diagnostics(L))
    factor = _timed_call(() -> cholesky(K))
    mass_basis = _timed_call(() ->
        CYAxiverse.generate.leading_hessian_mass_basis_float64(
            K, selected.value.Ltilde, selected.value.Qtilde))
    branches = _timed_call(() ->
        CYAxiverse.generate.leading_critical_branches(
            selected.value; max_branches))
    classified = _timed_call(() -> _classify_branches(
        branches.value, Q, L, factor.value))

    masses, mass_signs, _ = mass_basis.value
    least_tachyonic = classified.value.least_tachyonic
    best = classified.value.best
    flattest = classified.value.flattest
    (; h11=geom_idx.h11, polytope=geom_idx.polytope, frst=geom_idx.frst,
       status=:success, instantons=size(Q, 2), selected_instantons=size(
           selected.value.Qtilde, 2), qtilde_det=abs(det(Float64.(selected.value.Qtilde))),
       leading_log_gap=hierarchy.value.leading_log_gap,
       log_scale_span=hierarchy.value.log_scale_span,
       strong_hierarchy=hierarchy.value.heuristic_strong_hierarchy,
       branch_count=branches.value.branch_count,
       leading_minima_count=branches.value.leading_minima_count,
       saddle_count=classified.value.saddle_count,
       candidate_slowroll_saddles=classified.value.candidate_count,
       least_tachyonic_min_eta=least_tachyonic === nothing ? NaN : least_tachyonic.min_eta,
       least_tachyonic_epsilon=least_tachyonic === nothing ? NaN : least_tachyonic.epsilon,
       best_min_eta=best === nothing ? NaN : best.min_eta,
       best_epsilon=best === nothing ? NaN : best.epsilon,
       best_abs_min_eta=flattest === nothing ? NaN : flattest.abs_min_eta,
       mass_min=minimum(masses), mass_max=maximum(masses),
       negative_mass_count=count(<(0), mass_signs),
       stage_load_s=loaded.seconds, stage_select_s=selected.seconds,
       stage_hierarchy_s=hierarchy.seconds, stage_factor_s=factor.seconds,
       stage_mass_basis_s=mass_basis.seconds, stage_branches_s=branches.seconds,
       stage_classify_s=classified.seconds,
       stage_allocated_bytes=loaded.bytes + selected.bytes + hierarchy.bytes +
           factor.bytes + mass_basis.bytes + branches.bytes + classified.bytes,
       total_seconds=(time_ns() - started) / 1e9)
end

function main(args)
    options = _parse_args(args)
    ENV["CYAXIVERSE_DATA_DIR"] = options.data_dir
    for geom_idx in options.geometries
        summary = try
            run_geometry(geom_idx; max_branches=options.max_branches)
        catch error
            message = sprint(showerror, error)
            status = error isa ArgumentError &&
                occursin("leading branch enumeration would create", message) ?
                :branch_cap : :failed
            (; h11=geom_idx.h11, polytope=geom_idx.polytope, frst=geom_idx.frst,
               status, error=message)
        end
        println(summary)
        flush(stdout)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
