#!/usr/bin/env julia

"""Create and inspect deterministic fixtures for the inflation comparison."""

using CYAxiverse
using LinearAlgebra
using Serialization
using Printf

const Bench = CYAxiverse.axion_benchmarks
const Poly102 = Bench.poly102_inflation
const Source = Bench.benchmark_source_identity()
const Manifest = Bench.benchmark_manifest()

function n8_fixture(delta_k)
    k = Bench.N8_KC + delta_k
    potential = Poly102.n8_potential(k=k; trajectory=true)
    critical = Poly102.n8_degenerate_point()
    initial = Poly102.n8_inflation_initial_condition(k)
    derivatives = Poly102.n8_potential_derivatives(critical.theta, k; trajectory=true)
    kinetic = Matrix(Poly102.n8_kinetic_matrix(k))
    maps = Poly102.n8_coordinate_maps(k)
    initial_derivatives = Poly102.n8_potential_derivatives(
        initial.theta, k; trajectory=true)
    initial_canonical_gradient = maps.canonical_to_raw' * initial_derivatives.gradient
    probe = Poly102.n8_hilltop_normal_form(delta_k; sample_count=20)
    catastrophe = Bench.n8_catastrophe_diagnostic(
        theta=critical.theta, k=critical.k; trajectory=true)
    catastrophe_high_precision = Bench.n8_catastrophe_diagnostic(
        theta=critical.theta, k=critical.k; trajectory=true,
        precision_bits=120, tolerance=1e-20, derivative_tolerance=1e-12)
    audit = Poly102.n8_basis_directions(k)
    basis_initial_conditions = Dict{Symbol,Any}(
       label => Poly102.n8_inflation_initial_condition(k; direction_raw=direction)
       for (label, direction) in pairs(audit.directions)
    )
    basis_initial_conditions[:E_mass_eigenbasis] =
       Poly102.n8_inflation_initial_condition(k; basis=:mass_eigenbasis)
    basis_probes = Dict{Symbol,Any}(
       label => Poly102.n8_hilltop_normal_form(delta_k; direction_raw=direction)
       for (label, direction) in pairs(audit.directions)
    )
    basis_probes[:E_mass_eigenbasis] =
       Poly102.n8_hilltop_normal_form(delta_k; basis=:mass_eigenbasis)
    (; example=:n8_poly102, source_identity=Source, source_manifest=Manifest,
       scale_status=Manifest.volume_scaling.scale_status,
       claim_boundary=Manifest.claim_boundary,
       Q=potential.Q, phases=potential.phases,
       tau=potential.qdotτ,
       divisor_volumes=Bench.n8_geometry().divisor_volumes .* k,
       V_CY=126.0 * k^(3 / 2), k, kc=Bench.N8_KC,
       amplitudes=derivatives.amplitudes, K=kinetic,
       critical_point=critical.theta, hessian=derivatives.hessian,
       initial_point=initial.theta, initial_tangent=initial.initial_tangent,
       initial_gradient=initial_derivatives.gradient,
       initial_canonical_gradient,
       initial_vector_field=-initial_canonical_gradient / initial_derivatives.value,
       canonical_norm=initial.canonical_norm,
       end_thresholds=(eta=1.0, epsilon=1.0),
       solver=(rtol=1e-6, atol=1e-9, max_step=5.0, tmax=1e6),
       phase_input=Bench.phase_fixture(:n8; trajectory=true),
       catastrophe, catastrophe_high_precision,
       trajectory_probe=probe,
       trajectory_observables=Bench.trajectory_observables(probe),
       basis_audit=(directions=audit.directions, overlap=audit.overlap,
           metric_eigenvalues=audit.metric_eigenvalues,
           draft_kinetic_index=audit.draft_kinetic_index,
           canonical_hessian=audit.canonical_hessian,
           mass_eigenbasis=audit.mass_eigenbasis,
           equivalent_mass_direction=audit.equivalent_mass_direction,
           initial_conditions=basis_initial_conditions, probes=basis_probes),
       independent_physical_flow=(
           source=:miniforge_python_scipy,
           event_policy=:final_finite_exit,
           tuned=(delta_k=1e-7, efolds=464213.5708051051),
           sixty_efolds=(delta_k=1.5320548620798324e-3,
               efolds=59.690642055250756),
       ),
       reference=Poly102.n8_hilltop_normal_form_efolds(delta_k).efolds)
end

function n5_fixture(delta_k)
    raw = Poly102.n5_potential(k=Poly102.n5_critical_scale() + delta_k)
    k = Poly102.n5_critical_scale() + delta_k
    ratio = Poly102.n5_reduced_ratio(k)
    geometry = Poly102.n5_geometry()
    light = Poly102.n5_light_direction(k)
    catastrophe = Bench.n5_catastrophe_diagnostic(k=Poly102.n5_critical_scale())
    catastrophe_high_precision = Bench.n5_catastrophe_diagnostic(
        k=Poly102.n5_critical_scale(), precision_bits=120, tolerance=1e-20)
    (; example=:n5, source_identity=Source, source_manifest=Manifest,
        scale_status=Manifest.volume_scaling.scale_status,
        claim_boundary=Manifest.claim_boundary,
        Q=reshape(Int[1, 2], 1, 2), phases=zeros(2),
       tau=Float64[31.875, 32.0],
       divisor_volumes=geometry.divisor_volumes .* k,
       raw_Q=raw.Q, raw_tau=raw.qdotτ,
       V_CY=geometry.volume * k^(3 / 2), k,
       kc=Poly102.n5_critical_scale(),
       amplitudes=Float64[1.0, ratio], K=Matrix(Poly102.n5_kinetic_matrix(k)),
       geometry, light_direction=light.direction,
       light_charge_projections=Poly102.N5_Q * Poly102.N5_LIGHT_DIRECTION,
       critical_point=[π], hessian=[-1 + 4ratio],
       initial_point=[π - 1e-8], initial_tangent=[1.0],
       canonical_norm=1e-8,
       end_thresholds=(eta=1.0, epsilon=1.0),
       solver=(rtol=1e-6, atol=1e-9, max_step=5.0, tmax=1e6),
       phase_input=Bench.phase_fixture(:n5),
       catastrophe, catastrophe_high_precision,
       reference=Poly102.n5_hilltop_normal_form_efolds(delta_k).efolds)
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
                volume_scaling=Manifest.volume_scaling,
                scalar_amplitude="delta_H = sqrt(V)/(5*sqrt(6*pi*epsilon))",
                scalar_amplitude_convention=:paper_delta_H,
                mass_basis="H_theta * v = m² * K * v, fixed at the hilltop",
                trajectory_basis="do not recompute mass eigenvectors along the path",
            ),
        ))
    end
    path
end

function write_comparison(path)
    reference = Bench.benchmark_efold_targets()
    independent_physical_flow = Dict(
        1e-7 => 464213.5708051051,
        1.5320548620798324e-3 => 59.690642055250756,
    )
    rows = (
        ("N5", 1e-7, Poly102.n5_hilltop_normal_form_efolds(1e-7).efolds,
            reference.n5[1].efolds, "", reference.n5[1].efolds,
            "local reduced model; no independent author trajectory"),
        ("N5", 6.65e-5, Poly102.n5_hilltop_normal_form_efolds(6.65e-5).efolds,
            reference.n5[2].efolds, "", reference.n5[2].efolds,
            "local reduced model; no independent author trajectory"),
        ("N8_poly102", 1e-7, Bench.n8_local_hilltop_efolds(1e-7).efolds,
            reference.n8[1].efolds, independent_physical_flow[1e-7],
            reference.n8[1].efolds,
            "local normal form; miniforge Python physical flow"),
        ("N8_poly102", 1.5320548620798324e-3,
            Bench.n8_local_hilltop_efolds(1.5320548620798324e-3).efolds,
            reference.n8[2].efolds,
            independent_physical_flow[1.5320548620798324e-3],
            reference.n8[2].efolds,
            "local normal form; miniforge Python physical flow"),
    )
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "benchmark,delta_k,reduced_model_efolds,normal_form_efolds,independent_physical_flow_efolds,reference_efolds,method")
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
            Poly102.n8_hilltop_normal_form(1e-7; basis=:mass_eigenbasis)
        else
            Poly102.n8_hilltop_normal_form(1e-7; direction_raw=direction)
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
            ("N=5 delta=1e-7", Poly102.n5_hilltop_normal_form_efolds(1e-7)),
            ("N=5 delta=6.65e-5", Poly102.n5_hilltop_normal_form_efolds(6.65e-5)),
            ("N=8 delta=1e-7", Poly102.n8_hilltop_normal_form_efolds(1e-7)),
            ("N=8 delta=1.5320548620798324e-3",
             Poly102.n8_hilltop_normal_form_efolds(1.5320548620798324e-3)),
        )
        @printf("%s  N_e=%.10g\n", label, result.efolds)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
