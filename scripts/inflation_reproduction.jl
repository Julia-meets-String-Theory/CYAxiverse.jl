#!/usr/bin/env julia

"""Create and inspect deterministic fixtures for the inflation comparison."""

using CYAxiverse
using LinearAlgebra
using Serialization
using Printf

const Bench = CYAxiverse.paper_benchmarks

function n8_fixture(delta_k)
    k = Bench.N8_KC + delta_k
    potential = Bench.n8_potential(k=k; trajectory=true)
    critical = Bench.n8_degenerate_point()
    initial = Bench.n8_inflation_initial_condition(k)
    derivatives = Bench.n8_potential_derivatives(critical.theta, k; trajectory=true)
    kinetic = Matrix(Bench.n8_kinetic_matrix(k))
    maps = Bench.n8_coordinate_maps(k)
    initial_derivatives = Bench.n8_potential_derivatives(
        initial.theta, k; trajectory=true)
    initial_canonical_gradient = maps.canonical_to_raw' * initial_derivatives.gradient
    probe = Bench.n8_hilltop_probe(delta_k; sample_count=20)
    audit = Bench.n8_basis_directions(k)
    basis_initial_conditions = Dict{Symbol,Any}(
       label => Bench.n8_inflation_initial_condition(k; direction_raw=direction)
       for (label, direction) in pairs(audit.directions)
    )
    basis_initial_conditions[:E_mass_eigenbasis] =
       Bench.n8_inflation_initial_condition(k; basis=:mass_eigenbasis)
    basis_probes = Dict{Symbol,Any}(
       label => Bench.n8_hilltop_probe(delta_k; direction_raw=direction)
       for (label, direction) in pairs(audit.directions)
    )
    basis_probes[:E_mass_eigenbasis] =
       Bench.n8_hilltop_probe(delta_k; basis=:mass_eigenbasis)
    (; example=:n8_poly102, Q=potential.Q, phases=potential.phases,
       tau=potential.qdotτ, V_CY=126.0, k, kc=Bench.N8_KC,
       amplitudes=derivatives.amplitudes, K=kinetic,
       critical_point=critical.theta, hessian=derivatives.hessian,
       initial_point=initial.theta, initial_tangent=initial.initial_tangent,
       initial_gradient=initial_derivatives.gradient,
       initial_canonical_gradient,
       initial_vector_field=-initial_canonical_gradient / initial_derivatives.value,
       canonical_norm=initial.canonical_norm,
       end_thresholds=(eta=1.0, epsilon=1.0),
       solver=(rtol=1e-6, atol=1e-9, max_step=5.0, tmax=1e6),
       trajectory_probe=probe,
       basis_audit=(directions=audit.directions, overlap=audit.overlap,
           metric_eigenvalues=audit.metric_eigenvalues,
           draft_kinetic_index=audit.draft_kinetic_index,
           canonical_hessian=audit.canonical_hessian,
           mass_eigenbasis=audit.mass_eigenbasis,
           equivalent_mass_direction=audit.equivalent_mass_direction,
           initial_conditions=basis_initial_conditions, probes=basis_probes),
       author_independent=(
           source=:miniforge_python_scipy,
           event_policy=:final_finite_exit,
           tuned=(delta_k=1e-7, efolds=464213.5708051051),
           sixty_efolds=(delta_k=1.5320548620798324e-3,
               efolds=59.690642055250756),
       ),
       reference=Bench.n8_hilltop_efolds(delta_k).efolds)
end

function n5_fixture(delta_k)
    raw = Bench.n5_potential(k=Bench.n5_critical_scale() + delta_k)
    k = Bench.n5_critical_scale() + delta_k
    ratio = Bench.n5_reduced_ratio(k)
    geometry = Bench.n5_geometry()
    light = Bench.n5_light_direction(k)
    (; example=:n5, Q=reshape(Int[1, 2], 1, 2), phases=zeros(2),
       tau=Float64[31.875, 32.0], raw_Q=raw.Q, raw_tau=raw.qdotτ,
       V_CY=geometry.volume * k^(3 / 2), k,
       kc=Bench.n5_critical_scale(),
       amplitudes=Float64[1.0, ratio], K=Matrix(Bench.n5_kinetic_matrix(k)),
       geometry, light_direction=light.direction,
       light_charge_projections=Bench.N5_Q * Bench.N5_LIGHT_DIRECTION,
       critical_point=[π], hessian=[-1 + 4ratio],
       initial_point=[π - 1e-8], initial_tangent=[1.0],
       canonical_norm=1e-8,
       end_thresholds=(eta=1.0, epsilon=1.0),
       solver=(rtol=1e-6, atol=1e-9, max_step=5.0, tmax=1e6),
       reference=Bench.n5_hilltop_efolds(delta_k).efolds)
end

function write_fixture(path)
    mkpath(dirname(path))
    open(path, "w") do io
        serialize(io, Dict(
            :n5 => n5_fixture(1e-7),
            :n8_poly102 => n8_fixture(1e-7),
            :contract => (
                theta=(:raw_angle, :radian, :coordinate_vector),
                chi=(:canonical, :M_Pl, :coordinate_vector),
                tangent=(:raw_angle, :radian, :physical_tangent),
                potential="sum(Aᵢ * (1 - cos(Qᵢ⋅theta + phaseᵢ)))",
                metric="G(k) = G(kc) * (kc/k)^2",
                mass_basis="H_theta * v = m² * K * v, fixed at the hilltop",
                trajectory_basis="do not recompute mass eigenvectors along the path",
            ),
        ))
    end
    path
end

function write_comparison(path)
    reference = Bench.reference_efolds()
    author_independent = Dict(
        1e-7 => 464213.5708051051,
        1.5320548620798324e-3 => 59.690642055250756,
    )
    rows = (
        ("N5", 1e-7, Bench.n5_hilltop_efolds(1e-7).efolds,
            reference.n5[1].efolds, "", reference.n5[1].efolds,
            "local reduced model; no independent author trajectory"),
        ("N5", 6.65e-5, Bench.n5_hilltop_efolds(6.65e-5).efolds,
            reference.n5[2].efolds, "", reference.n5[2].efolds,
            "local reduced model; no independent author trajectory"),
        ("N8_poly102", 1e-7, Bench.n8_hilltop_efolds(1e-7).efolds,
            reference.n8[1].efolds, author_independent[1e-7],
            reference.n8[1].efolds,
            "local normal form; miniforge Python physical flow"),
        ("N8_poly102", 1.5320548620798324e-3,
            Bench.n8_hilltop_efolds(1.5320548620798324e-3).efolds,
            reference.n8[2].efolds,
            author_independent[1.5320548620798324e-3],
            reference.n8[2].efolds,
            "local normal form; miniforge Python physical flow"),
    )
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "benchmark,delta_k,package_local_efolds,author_efolds,author_independent_efolds,draft_efolds,method")
        for row in rows
            println(io, join(row, ","))
        end
    end
    path
end

function print_basis_audit()
    k = Bench.N8_KC + 1e-7
    audit = Bench.n8_basis_directions(k)
    directions = audit.directions
    println("N=8/poly-102 basis audit")
    println("metric eigenvalues = ", audit.metric_eigenvalues)
    for (label, direction) in pairs(directions)
        norm_value = sqrt(dot(direction, audit.metric * direction))
        overlap = audit.overlap[findfirst(==(label), keys(directions)),
            findfirst(==(:B_package_current), keys(directions))]
        probe = if label === :E_mass_eigenbasis
            Bench.n8_hilltop_probe(1e-7; basis=:mass_eigenbasis)
        else
            Bench.n8_hilltop_probe(1e-7; direction_raw=direction)
        end
        @printf("%-22s canonical_norm=%.16g overlap=%.16g N_e=%.16g event=%s\n",
            label, norm_value, overlap, probe.efolds, probe.end_event)
        println("  initial theta = ", probe.initial.theta)
        println("  initial tangent = ", probe.initial.initial_tangent)
        @printf("  initial epsilon=%.16g eta_parallel=%.16g samples=%d\n",
            probe.initial.epsilon, probe.initial.eta_parallel, length(probe.samples))
    end
    println("canonical/mass overlap = ", audit.equivalent_mass_direction)
end

function main()
    output = isempty(ARGS) ? joinpath(@__DIR__, "..", "validation", "inflation_fixtures.jls") : abspath(ARGS[1])
    write_fixture(output)
    write_comparison(joinpath(dirname(output), "inflation_comparison.csv"))
    println("fixture = ", output)
    print_basis_audit()
    for (label, result) in (
            ("N=5 delta=1e-7", Bench.n5_hilltop_efolds(1e-7)),
            ("N=5 delta=6.65e-5", Bench.n5_hilltop_efolds(6.65e-5)),
            ("N=8 delta=1e-7", Bench.n8_hilltop_efolds(1e-7)),
            ("N=8 delta=1.5320548620798324e-3",
             Bench.n8_hilltop_efolds(1.5320548620798324e-3)),
        )
        @printf("%s  N_e=%.10g\n", label, result.efolds)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
