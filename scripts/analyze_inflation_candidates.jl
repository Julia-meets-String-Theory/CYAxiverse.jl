#!/usr/bin/env julia

using CYAxiverse
using HDF5
using LinearAlgebra
using Printf
using Statistics

const GeometryIndex = CYAxiverse.structs.GeometryIndex

function derivatives(theta, Q, L)
    logscale = L[2, :]
    amplitudes = L[1, :] .* 10.0 .^ (logscale .- maximum(logscale))
    args = 2π .* (Q' * theta)
    value = sum(amplitudes .* (1 .- cos.(args)))
    gradient = 2π .* Q * (amplitudes .* sin.(args))
    weights = amplitudes .* cos.(args)
    hessian = 4π^2 .* Q * Diagonal(weights) * Q'
    (; value, gradient, hessian, amplitudes)
end

function classify_point(theta, Q, L, K)
    d = derivatives(theta, Q, L)
    factor = cholesky(Hermitian(K)).L
    to_theta = inv(factor')
    canonical_hessian = to_theta' * d.hessian * to_theta
    eigs = eigvals(Hermitian(canonical_hessian))
    kinetic_inverse = inv(K)
    gradnorm = sqrt(max(dot(d.gradient, kinetic_inverse * d.gradient), 0.0))
    epsilon = d.value == 0 ? Inf : 0.5 * (gradnorm / abs(d.value))^2
    eta_values = d.value == 0 ? fill(Inf, length(eigs)) : eigs ./ d.value
    (; value=d.value, gradnorm, epsilon,
       min_eta=minimum(eta_values), max_eta=maximum(eta_values),
       abs_min_eta=minimum(abs.(eta_values)),
       negative_modes=count(<(0), eigs),
       zeroish_modes=count(x -> abs(x) <= 1e-10 * max(maximum(abs, eigs), 1.0), eigs),
       positive_modes=count(>(0), eigs),
       eta_values, hessian_eigenvalues=eigs)
end

function reduced_solve(Q, L, selected; starts)
    branches = CYAxiverse.generate.leading_critical_branches(selected)
    (; coordinates=branches.coordinates,
       original_coordinates=branches.coordinates,
       critical_count=branches.branch_count,
       leading_negative_modes=branches.leading_negative_modes,
       minima_count=branches.leading_minima_count,
       starts=0,
       unique_original_count=branches.branch_count)
end

function analyze_geometry(geom_idx; starts=8192)
    oriented = CYAxiverse.read.oriented_potential(geom_idx)
    Q, L, K = oriented.Q, oriented.L, oriented.K
    selected = CYAxiverse.generate.LQtilde(Q, L)
    hierarchy = CYAxiverse.generate.instanton_hierarchy_diagnostics(L)
    solved = reduced_solve(Q, L, selected; starts=starts)

    point_rows = NamedTuple[]
    for i in axes(solved.original_coordinates, 2)
        theta = solved.original_coordinates[:, i]
        c = classify_point(theta, Q, L, K)
        push!(point_rows, (
            h11=geom_idx.h11,
            polytope=geom_idx.polytope,
            frst=geom_idx.frst,
            point_index=i,
            leading_negative_modes=solved.leading_negative_modes[i],
            value=c.value,
            gradnorm=c.gradnorm,
            epsilon=c.epsilon,
            min_eta=c.min_eta,
            max_eta=c.max_eta,
            abs_min_eta=c.abs_min_eta,
            negative_modes=c.negative_modes,
            zeroish_modes=c.zeroish_modes,
            positive_modes=c.positive_modes,
        ))
    end

    saddle_rows = filter(row -> row.negative_modes > 0 && row.value > 0, point_rows)
    best = isempty(saddle_rows) ? nothing :
        saddle_rows[argmin([abs(row.min_eta + 1) for row in saddle_rows])]
    flattest = isempty(saddle_rows) ? nothing :
        saddle_rows[argmin([abs(row.min_eta) for row in saddle_rows])]
    least_tachyonic = isempty(saddle_rows) ? nothing :
        saddle_rows[argmax([row.min_eta for row in saddle_rows])]

    summary = (
        h11=geom_idx.h11,
        polytope=geom_idx.polytope,
        frst=geom_idx.frst,
        qtilde_det=abs(round(Int, det(selected.Qtilde))),
        instantons=size(Q, 2),
        enumeration="leading_half_integer_branches",
        starts_used=solved.starts,
        branch_count=solved.critical_count,
        unique_original_count=solved.unique_original_count,
        leading_minima_count=solved.minima_count,
        saddle_count=length(saddle_rows),
        leading_log_gap=hierarchy.leading_log_gap,
        log_scale_span=hierarchy.log_scale_span,
        least_tachyonic_min_eta=least_tachyonic === nothing ? NaN : least_tachyonic.min_eta,
        least_tachyonic_epsilon=least_tachyonic === nothing ? NaN : least_tachyonic.epsilon,
        least_tachyonic_value=least_tachyonic === nothing ? NaN : least_tachyonic.value,
        best_min_eta=best === nothing ? NaN : best.min_eta,
        best_abs_min_eta=flattest === nothing ? NaN : flattest.abs_min_eta,
        best_epsilon=flattest === nothing ? NaN : flattest.epsilon,
        candidate_slowroll_saddles=count(row -> row.epsilon < 1 && abs(row.min_eta) < 1, saddle_rows),
    )
    summary, point_rows
end

function csv_field(x)
    if x isa AbstractFloat
        return isfinite(x) ? string(x) : string(x)
    end
    s = string(x)
    if occursin(r"[,\n\"]", s)
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    s
end

function write_namedtuple_csv(path, rows)
    open(path, "w") do io
        if isempty(rows)
            return
        end
        names = propertynames(first(rows))
        println(io, join(string.(names), ","))
        for row in rows
            println(io, join((csv_field(getproperty(row, name)) for name in names), ","))
        end
    end
end

function parse_geometries(args)
    if isempty(args)
        return GeometryIndex[
            GeometryIndex(5, 1, 1),
            GeometryIndex(9, 1, 1),
            GeometryIndex(10, 1, 1),
            GeometryIndex(11, 1, 1),
            GeometryIndex(11, 2, 1),
            GeometryIndex(11, 7, 1),
        ]
    end
    geoms = GeometryIndex[]
    for arg in args
        parts = parse.(Int, split(arg, ","))
        length(parts) == 3 || error("geometry arguments must be h11,polytope,frst")
        push!(geoms, GeometryIndex(parts...))
    end
    geoms
end

function main(args)
    ENV["CYAXIVERSE_DATA_DIR"] = get(ENV, "CYAXIVERSE_DATA_DIR",
        "/Users/vmehta/Documents/CYAxiverse/cyaxiverse/data")
    outdir = joinpath(@__DIR__, "..", "paper_benchmarks", "2023_minima", "inflation_screen")
    mkpath(outdir)
    summaries = NamedTuple[]
    all_points = NamedTuple[]
    starts = parse(Int, get(ENV, "CYAXIVERSE_INFLATION_SCREEN_STARTS", "8192"))

    for geom_idx in parse_geometries(args)
        @printf("analyzing h11=%d polytope=%d frst=%d starts=%d\n",
            geom_idx.h11, geom_idx.polytope, geom_idx.frst, starts)
        summary, points = analyze_geometry(geom_idx; starts=starts)
        push!(summaries, summary)
        append!(all_points, points)
        @printf("  branches=%d leading_minima=%d saddles=%d slowroll_saddles=%d least_tachyonic_eta=%.4g\n",
            summary.branch_count, summary.leading_minima_count, summary.saddle_count,
            summary.candidate_slowroll_saddles, summary.least_tachyonic_min_eta)
    end

    write_namedtuple_csv(joinpath(outdir, "candidate_summary.csv"), summaries)
    write_namedtuple_csv(joinpath(outdir, "candidate_critical_points.csv"), all_points)
    println(joinpath(outdir, "candidate_summary.csv"))
    println(joinpath(outdir, "candidate_critical_points.csv"))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
