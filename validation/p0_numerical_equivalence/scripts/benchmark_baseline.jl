#!/usr/bin/env julia

"""Bounded P0 B2--B7 baseline for the pinned CYAxiverse source.

This is an evidence harness, not a production implementation.  It uses the
current public/module-level routes and deterministic in-memory or synthetic
fixtures.  It intentionally does not run a geometry population scan.
"""

using CYAxiverse
using LinearAlgebra
using Random
using SHA
using Statistics
using Dates
using Pkg

const G = CYAxiverse.generate
const PB = CYAxiverse.paper_benchmarks
const IP = CYAxiverse.inflation_points
const FS = CYAxiverse.filestructure
const RD = CYAxiverse.read
const H5 = CYAxiverse.read.HDF5
const GeometryIndex = CYAxiverse.structs.GeometryIndex

const ROOT = normpath(joinpath(@__DIR__, ".."))
const OUT = joinpath(ROOT, "benchmark_results")

"""A compact, stable digest for a fixture's numerical identity."""
fixture_digest(parts...) = bytes2hex(sha256(codeunits(join(string.(parts, "|")))))

"""Return a short one-line summary without serializing large arrays."""
function short_summary(x)
    x === nothing && return "nothing"
    x isa AbstractArray && return "size=$(size(x)), eltype=$(eltype(x)), first=$(isempty(x) ? "empty" : repr(first(x)))"
    x isa Number && return repr(x)
    x isa Symbol && return string(x)
    string(x)
end

"""Render a diagnostic value on one TSV-safe line, retaining full arrays."""
function diagnostic_repr(x)
    x === nothing && return "unavailable"
    value = repr(x)
    replace(value, '\t' => ' ', '\n' => ' ', '\r' => ' ')
end

"""Return the deterministic Halton starts used by `critical_points` here."""
function critical_start_set(starts::Int)
    starts > 0 || throw(ArgumentError("starts must be positive"))
    starts_matrix = Matrix{Float64}(undef, 2, starts)
    starts_matrix[:, 1] .= 0.0
    for sample in 2:starts
        index = sample - 1
        for (row, base) in enumerate((2, 3))
            result = 0.0
            factor = inv(Float64(base))
            remaining = index
            while remaining > 0
                remaining, digit = divrem(remaining, base)
                result += digit * factor
                factor /= base
            end
            starts_matrix[row, sample] = result
        end
    end
    starts_matrix
end

"""Distance on the unit torus used by `critical_points` deduplication."""
periodic_distance(left, right) =
    maximum(min.(abs.(left .- right), 1 .- abs.(left .- right)))

"""One-to-one root matching witnesses for two deterministic replays."""
function root_matching_witnesses(left, right)
    size(left, 1) == size(right, 1) ||
        throw(DimensionMismatch("root sets have different dimensions"))
    size(left, 2) == size(right, 2) ||
        throw(DimensionMismatch("root sets have different cardinalities"))
    used = falses(size(right, 2))
    witnesses = NamedTuple[]
    for i in axes(left, 2)
        candidates = [(periodic_distance(left[:, i], right[:, j]), j)
            for j in axes(right, 2) if !used[j]]
        isempty(candidates) && throw(ErrorException("one-to-one matching failed"))
        distance, j = candidates[argmin(first.(candidates))]
        used[j] = true
        push!(witnesses, (; left_index=i, right_index=j, periodic_distance=distance))
    end
    witnesses
end

"""Compact correction diagnostics, including the final coordinate."""
function correction_summary(c, diagnostics)
    c === nothing && return "unavailable"
    diagnostic = diagnostics === nothing ? "unavailable" :
        "value=$(diagnostic_repr(diagnostics.value)), gradient_norm=$(diagnostic_repr(diagnostics.gradient_norm)), gradient_residual=$(diagnostic_repr(diagnostics.gradient_residual)), epsilon=$(diagnostic_repr(diagnostics.epsilon)), eta_values=$(diagnostic_repr(diagnostics.eta_values)), hessian_eigenvalues=$(diagnostic_repr(diagnostics.hessian_eigenvalues)), inertia=($(diagnostics.negative_modes),$(diagnostics.zeroish_modes),$(diagnostics.positive_modes)), zero_tolerance=$(diagnostic_repr(diagnostics.zero_tolerance)), physical_basis=$(diagnostics.physical_basis)"
    "status=$(c.status), theta=$(diagnostic_repr(c.theta)), residual=$(diagnostic_repr(c.residual)), iterations=$(c.iterations), seconds=$(c.seconds), error=$(diagnostic_repr(c.error)), working_basis=$(c.working_basis), diagnostics=[$diagnostic]"
end

"""Full state diagnostics exposed by the bounded N=8 trajectory API."""
function trajectory_summary(result; include_final_unavailable=true)
    result === nothing && return "unavailable"
    initial = result.initial
    saved_samples = diagnostic_repr(result.samples)
    final_state = include_final_unavailable ?
        "unavailable_from_current_api" : "saved_endpoint_sample"
    "initial_theta=$(diagnostic_repr(initial.theta)), initial_epsilon=$(diagnostic_repr(initial.epsilon)), initial_eta_parallel=$(diagnostic_repr(initial.eta_parallel)), initial_gradient_norm=$(norm(initial.initial_gradient)), initial_basis=$(initial.basis), initial_displacement=$(initial.displacement), saved_samples=$(saved_samples), final_state=$(final_state)"
end

"""Full spectrum diagnostics and quartic log arrays exposed by the API."""
function spectrum_summary(s, kinetic, lq; precision_label)
    s === nothing && return "unavailable"
    float_basis = precision_label == :float64 ?
        G.leading_hessian_mass_basis_float64(kinetic, lq.Ltilde, lq.Qtilde) :
        G.leading_hessian_mass_basis(kinetic, lq.Ltilde, lq.Qtilde; prec=80)
    masses, signs, eigenvectors = float_basis
    "status=returned_AxionSpectrum, precision=$(precision_label), m=$(diagnostic_repr(s.m)), msign=$(diagnostic_repr(s.msign)), f=$(diagnostic_repr(s.f)), fK=$(diagnostic_repr(s.fK)), eigenvectors_api=leading_hessian_mass_basis, eigenvectors=$(diagnostic_repr(eigenvectors)), basis_m=$(diagnostic_repr(masses)), basis_msign=$(diagnostic_repr(signs)), lambda_self_sign=$(diagnostic_repr(s.λselfsign)), lambda_self_log10=$(diagnostic_repr(s.λself)), lambda31_indices=$(diagnostic_repr(s.λ31_i)), lambda31_sign=$(diagnostic_repr(s.λ31sign)), lambda31_log10=$(diagnostic_repr(s.λ31)), lambda22_indices=$(diagnostic_repr(s.λ22_i)), lambda22_sign=$(diagnostic_repr(s.λ22sign)), lambda22_log10=$(diagnostic_repr(s.λ22)), mass_basis_diagnostics=$(diagnostic_repr(s.mass_basis_diagnostics)), quartic_diagnostics=$(diagnostic_repr(s.quartic_diagnostics)), instanton_hierarchy=$(diagnostic_repr(s.instanton_hierarchy))"
end

"""Measure a warmed function with explicit allocation accounting.

One call is discarded as warmup.  Each measured sample runs after `GC.gc()`.
`@timed.bytes` is the primary allocation statistic and one post-warmup
`@allocated` call is retained as a cross-check.
"""
function measure(name::AbstractString, f; samples::Int=5, summary::Function=() -> "")
    samples > 0 || throw(ArgumentError("samples must be positive"))
    f()                         # warmup / compilation
    GC.gc()
    times = Float64[]
    bytes = Int[]
    gc_times = Float64[]
    for _ in 1:samples
        GC.gc()
        timed = @timed f()
        push!(times, timed.time)
        push!(bytes, timed.bytes)
        push!(gc_times, timed.gctime)
    end
    GC.gc()
    allocated_crosscheck = @allocated f()
    result_summary = try
        summary()
    catch err
        "summary_error=$(sprint(showerror, err))"
    end
    (; name=String(name), samples, median_seconds=median(times),
       minimum_seconds=minimum(times), maximum_seconds=maximum(times),
       median_bytes=Int(round(median(bytes))),
       allocated_crosscheck, median_gc_seconds=median(gc_times),
       summary=replace(result_summary, '\t' => ' ', '\n' => ' ', '\r' => ' '),
       status="completed")
end

function failed_row(name, err)
    (; name=String(name), samples=0, median_seconds=NaN,
       minimum_seconds=NaN, maximum_seconds=NaN, median_bytes=-1,
       allocated_crosscheck=-1, median_gc_seconds=NaN,
       summary=replace(sprint(showerror, err), '\t' => ' '),
       status="failed:$(typeof(err))")
end

function safe_measure(rows, name, f; kwargs...)
    try
        push!(rows, measure(name, f; kwargs...))
    catch err
        push!(rows, failed_row(name, err))
    end
end

function write_rows(path, rows)
    open(path, "w") do io
        println(io, "name\tsamples\tmedian_seconds\tminimum_seconds\tmaximum_seconds\tmedian_bytes\tallocated_crosscheck\tmedian_gc_seconds\tstatus\tsummary")
        for row in rows
            println(io, join((row.name, row.samples,
                row.median_seconds, row.minimum_seconds, row.maximum_seconds,
                row.median_bytes, row.allocated_crosscheck,
                row.median_gc_seconds, row.status, row.summary), '\t'))
        end
    end
end

function write_metadata(path; kwargs...)
    open(path, "w") do io
        for (key, value) in kwargs
            println(io, key, "=", value)
        end
    end
end

function direct_fixtures()
    # The paper benchmark stores Q as axions × instantons and L as 2 × N.
    p5 = PB.n5_potential(k=1.0)
    p8 = PB.n8_potential(k=1.0)
    l5raw = hcat(vec(p5.L[1, :]), vec(p5.L[2, :]))
    l8raw = hcat(vec(p8.L[1, :]), vec(p8.L[2, :]))
    x5 = Float64[0.11, -0.07, 0.03, 0.19, -0.13]
    x8 = Float64[0.017, -0.023, 0.031, -0.041, 0.053, -0.067, 0.079, -0.089]
    (; p5, p8, l5raw, l8raw, x5, x8)
end

function run_b2!(rows, fixtures, identities)
    p = fixtures.p8
    x = fixtures.x8
    q = p.Q
    lraw = fixtures.l8raw
    identities["B2_dense_n8"] = fixture_digest("B2", q, p.L, x)
    # These are the historical dense routes in generate.jl (Q: h11 × N,
    # L: N × (sign, log10 scale)); the paper route is recorded separately.
    safe_measure(rows, "B2_dense_generate_value_n8",
        () -> G.V(x, lraw, q); summary=() -> short_summary(G.V(x, lraw, q)))
    safe_measure(rows, "B2_dense_generate_gradient_n8",
        () -> G.jacobian(x, lraw, q); summary=() -> short_summary(G.jacobian(x, lraw, q)))
    safe_measure(rows, "B2_dense_generate_value_gradient_n8",
        () -> (G.V(x, lraw, q), G.jacobian(x, lraw, q));
        summary=() -> "value=$(G.V(x, lraw, q)), gradient=$(short_summary(G.jacobian(x, lraw, q)))")
    safe_measure(rows, "B2_dense_generate_gradient_hessian_n8",
        () -> (G.jacobian(x, lraw, q), G.hessian(x, lraw, q));
        summary=() -> "gradient=$(short_summary(G.jacobian(x, lraw, q))), hessian=$(short_summary(G.hessian(x, lraw, q)))")
    safe_measure(rows, "B2_dense_generate_value_gradient_hessian_n8",
        () -> (G.V(x, lraw, q), G.jacobian(x, lraw, q), G.hessian(x, lraw, q));
        summary=() -> "value=$(G.V(x, lraw, q)), gradient=$(short_summary(G.jacobian(x, lraw, q))), hessian=$(short_summary(G.hessian(x, lraw, q)))")
    safe_measure(rows, "B2_dense_paper_derivatives_n8",
        () -> PB.n8_potential_derivatives(x, 1.0);
        summary=() -> begin
            d = PB.n8_potential_derivatives(x, 1.0)
            "value=$(d.value), gradient=$(short_summary(d.gradient)), hessian=$(short_summary(d.hessian))"
        end)
end

function structured_fixture(h11::Int)
    Random.seed!(20260915 + h11)
    # pseudo_Q emits instanton rows, so transpose to the package's canonical
    # h11 × N orientation.  The coefficient/log encoding is deterministic and
    # deliberately spans a bounded hierarchy without underflow.
    q = Matrix{Int}(G.pseudo_Q(h11, 1, 1)')
    n = size(q, 2)
    signs = [isodd(i) ? 1.0 : -1.0 for i in 1:n]
    logs = [-0.4 * (i - 1) for i in 1:n]
    l = vcat(reshape(signs, 1, :), reshape(logs, 1, :))
    theta = [0.017 * (i - (h11 + 1) / 2) for i in 1:h11]
    (; q, l, theta, base_count=h11 + 4,
       digest=fixture_digest("structured", h11, q, l, theta))
end

function run_b3!(rows, identities)
    for h11 in (4, 8)
        fixture = structured_fixture(h11)
        identities["B3_structured_h$(h11)"] = fixture.digest
        evaluator = G.structured_charge_evaluator(fixture.q, fixture.l;
            base_count=fixture.base_count)
        generic_workspace = G.logshifted_derivative_workspace(fixture.q, fixture.l)
        structured_value = Ref{Any}(nothing)
        generic_value = Ref{Any}(nothing)
        safe_measure(rows, "B3_preparation_structure_proof_h$(h11)",
            () -> G.structured_charge_evaluator(fixture.q, fixture.l;
                base_count=fixture.base_count);
            samples=3,
            summary=() -> "validated=$(evaluator.representation.validated), fallback=$(evaluator.uses_generic_fallback), base_count=$(fixture.base_count), instantons=$(size(fixture.q, 2))")
        safe_measure(rows, "B3_repeated_structured_evaluation_h$(h11)",
            () -> begin
                structured_value[] = G.structured_logshifted_derivatives!(
                    evaluator, fixture.theta, fixture.q)
                structured_value[]
            end;
            summary=() -> begin
                d = structured_value[]
                "validated=$(evaluator.representation.validated), value=$(d.value), gradient=$(short_summary(d.gradient)), hessian=$(short_summary(d.hessian))"
            end)
        safe_measure(rows, "B3_repeated_generic_evaluation_h$(h11)",
            () -> begin
                generic_value[] = G.logshifted_derivatives!(generic_workspace,
                    fixture.theta, fixture.q)
                generic_value[]
            end;
            summary=() -> begin
                d = generic_value[]
                "value=$(d.value), gradient=$(short_summary(d.gradient)), hessian=$(short_summary(d.hessian))"
            end)
        # Evaluate once outside timing and retain only a scalar parity check.
        structured = G.structured_logshifted_derivatives!(evaluator,
            fixture.theta, fixture.q)
        generic = G.logshifted_derivatives!(generic_workspace,
            fixture.theta, fixture.q)
        # The structured and generic loops have different summation orders.
        # Record parity at a predeclared Float64 roundoff envelope rather than
        # treating harmless last-bit differences as a scientific mismatch.
        parity = isapprox(structured.value, generic.value; rtol=1e-13, atol=1e-13) &&
            isapprox(structured.gradient, generic.gradient; rtol=1e-13, atol=1e-13) &&
            isapprox(structured.hessian, generic.hessian; rtol=1e-13, atol=1e-13)
        identities["B3_parity_h$(h11)"] = parity
    end
end

function run_b4!(rows, identities)
    # A small signed three-term two-axion potential keeps root enumeration
    # bounded while exercising periodic folding, deduplication and inertia.
    q = Int[1 0 1; 0 1 1]
    l = Float64[1.0 1.0 -1.0; 0.0 -0.2 -0.7]
    starts = 24
    start_set = critical_start_set(starts)
    identities["B4_critical_points_n2"] = fixture_digest("B4", q, l, starts, start_set)
    result = Ref{Any}(nothing)
    safe_measure(rows, "B4_critical_points_n2_starts24",
        () -> begin
            result[] = CYAxiverse.minimizer.critical_points(l, q;
                starts=starts, residual_tolerance=1e-10,
                merge_tolerance=1e-7, max_iterations=100)
            result[]
        end;
        samples=3,
        summary=() -> begin
            r = result[]
            replay = CYAxiverse.minimizer.critical_points(l, q;
                starts=starts, residual_tolerance=1e-10,
                merge_tolerance=1e-7, max_iterations=100)
            witnesses = root_matching_witnesses(r.coordinates, replay.coordinates)
            "starts=$(r.starts), deterministic_start_set=$(diagnostic_repr(start_set)), coordinates=$(diagnostic_repr(r.coordinates)), critical_count=$(r.critical_count), minima_count=$(r.minima_count), residuals=$(diagnostic_repr(r.residuals)), inertia=$(diagnostic_repr(r.inertia)), periodic_one_to_one_replay_witnesses=$(diagnostic_repr(witnesses)), status_availability=per_start_unavailable_public_api; returned_root_status=converged_only; replay_status=identical, public_fields=$(diagnostic_repr(propertynames(r)))"
        end)
end

function run_b5!(rows, fixtures, identities)
    # B5a: the bounded current fixed-step N=8 e-fold flow.  The current API
    # exposes accepted step count and exit event but not internal RHS/Hessian
    # counters; those fields remain explicitly unavailable in the report.
    delta_k = 1.0e-3
    identities["B5a_n8_slow_roll"] = fixture_digest("B5a", delta_k, 1e-6, 0.25, 0.1, 1.0e-3, 4, :canonical_hessian, PB.N8_BEST_X)
    flow = Ref{Any}(nothing)
    safe_measure(rows, "B5a_n8_slow_roll_trajectory_bounded",
        () -> begin
            flow[] = PB.n8_efold_gradient_flow(delta_k;
                displacement=1e-6, max_efolds=0.25, max_step=0.1,
                initial_step=1e-3, sample_count=4)
            flow[]
        end;
        samples=2,
        summary=() -> begin
            f = flow[]
            "entered=$(f.entered_slow_roll), end_event=$(f.end_event), steps=$(f.steps), efolds=$(f.efolds), samples=$(length(f.samples)), solver=$(f.solver), trajectory_diagnostics=$(trajectory_summary(f))"
        end)
    probe = Ref{Any}(nothing)
    safe_measure(rows, "B5a_n8_hilltop_probe_normal_form",
        () -> begin
            probe[] = PB.n8_hilltop_probe(delta_k;
                displacement=1e-6, sample_count=4)
            probe[]
        end;
        samples=3,
        summary=() -> begin
            p = probe[]
            "entered=$(p.entered_slow_roll), end_event=$(p.end_event), steps=$(p.steps), efolds=$(p.efolds), samples=$(length(p.samples)), trajectory_diagnostics=$(trajectory_summary(p; include_final_unavailable=false))"
        end)

    # B5b: stationary correction at Float64 plus a bounded BigFloat replay.
    p = fixtures.p5
    k = Matrix(PB.n5_kinetic_matrix(1.0))
    seed = Float64[0.02, -0.01, 0.03, -0.02, 0.01]
    high_residual_tolerance = 1e-40
    identities["B5b_stationary_correction_n5"] = fixture_digest("B5b", p.Q, p.L, k, seed, high_residual_tolerance)
    context = IP.prepare_context(p.Q, p.L, k)
    correction = Ref{Any}(nothing)
    safe_measure(rows, "B5b_stationary_correction_float64_n5",
        () -> begin
            correction[] = IP.correct_stationary_point(context, seed;
                residual_tolerance=1e-10, max_iterations=40,
                max_line_search=12)
            correction[]
        end;
        samples=3,
        summary=() -> begin
            c = correction[]
            "$(correction_summary(c, IP.diagnose(context, c.theta))), high_residual_tolerance=$(high_residual_tolerance)"
        end)
    comparison = Ref{Any}(nothing)
    safe_measure(rows, "B5b_stationary_correction_float64_bigfloat128_n5",
        () -> begin
            comparison[] = IP.compare_precision(seed, p.Q, p.L, k;
                precision_bits=128, float_residual_tolerance=1e-10,
                high_residual_tolerance=high_residual_tolerance, zero_tolerance=1e-10,
                max_iterations=40, max_line_search=12)
            comparison[]
        end;
        samples=1,
        summary=() -> begin
            c = comparison[]
            "accepted=$(c.accepted), residual_agreement=$(c.residual_agreement), inertia_agreement=$(c.inertia_agreement), high_residual_tolerance=$(high_residual_tolerance), float=$(correction_summary(c.float_correction, c.float_diagnostics)), high=$(correction_summary(c.high_correction, c.high_diagnostics))"
        end)
end

function run_b6!(rows, fixtures, identities)
    p = fixtures.p5
    k = p isa NamedTuple ? PB.n5_kinetic_matrix(1.0) : nothing
    identities["B6_pq_spectrum_n5"] = fixture_digest("B6", p.Q, p.L, k, :float64, :high_precision, 80, true, true, true)
    lq = G.LQtilde(p.Q, p.L)
    float_result = Ref{Any}(nothing)
    safe_measure(rows, "B6_pq_spectrum_float64_n5",
        () -> begin
            float_result[] = G.pq_spectrum(k, p.L, p.Q;
                mixing_correction=:float64, quartic_diagnostics=true,
                mass_basis_diagnostics=true,
                hierarchy_diagnostics=true)
            float_result[]
        end;
        samples=2,
        summary=() -> begin
            s = float_result[]
            "quartic_self_count=$(length(s.λself)), lambda31_count=$(length(s.λ31)), lambda22_count=$(length(s.λ22)), $(spectrum_summary(s, k, lq; precision_label=:float64))"
        end)
    high_result = Ref{Any}(nothing)
    safe_measure(rows, "B6_pq_spectrum_high_precision80_n5",
        () -> begin
            high_result[] = G.pq_spectrum(k, p.L, p.Q;
                mixing_correction=:high_precision, prec=80,
                quartic_diagnostics=true, mass_basis_diagnostics=true,
                hierarchy_diagnostics=true)
            high_result[]
        end;
        samples=1,
        summary=() -> begin
            s = high_result[]
            "quartic_self_count=$(length(s.λself)), lambda31_count=$(length(s.λ31)), lambda22_count=$(length(s.λ22)), $(spectrum_summary(s, k, lq; precision_label=:high_precision80))"
        end)
end

function write_hdf5_fixture(root)
    dir = joinpath(root, "h11_002", "np_0000001", "cy_0000001")
    mkpath(dir)
    path = joinpath(dir, "cyax.h5")
    q = Int[1 0 1; 0 1 1]
    l = Float64[1.0 1.0 -1.0; 0.0 -0.2 -0.7]
    kinv = Float64[1 0; 0 1]
    H5.h5open(path, "w") do file
        cytools = H5.create_group(file, "cytools")
        potential = H5.create_group(cytools, "potential")
        potential["Q"] = q
        potential["L"] = l
        geometric = H5.create_group(cytools, "geometric")
        geometric["Kinv"] = kinv
        geometric["h21"] = 1
        geometric["glsm"] = Int[1 0; 0 1]
        geometric["basis"] = Int[1, 2]
        geometric["tip"] = Float64[1.0, 1.0]
        geometric["CY_volume"] = 1.0
        geometric["divisor_volumes"] = Float64[1.0, 1.0]
    end
    (; path, q, l, kinv, digest=fixture_digest("B7", q, l, kinv))
end

function run_b7!(rows, identities)
    mktempdir() do root
        fixture = write_hdf5_fixture(root)
        identities["B7_hdf5_fixture"] = fixture.digest
        old_data = get(ENV, "CYAXIVERSE_DATA_DIR", nothing)
        old_args = get(ENV, "newARGS", nothing)
        ENV["CYAXIVERSE_DATA_DIR"] = root
        delete!(ENV, "newARGS")
        try
            idx = GeometryIndex(2, 1, 1)
            scan = Ref{Any}(nothing)
            safe_measure(rows, "B7_filesystem_scan_np_path_generate_h11_2",
                () -> begin
                    scan[] = FS.np_path_generate(2)
                    scan[]
                end;
                samples=3,
                summary=() -> "path_count=$(size(scan[][1], 2)), index_shape=$(size(scan[][2]))")
            query = Ref{Any}(nothing)
            safe_measure(rows, "B7_hdf5_query_oriented_potential_h11_2",
                () -> begin
                    query[] = RD.oriented_potential(idx)
                    query[]
                end;
                samples=3,
                summary=() -> begin
                    q = query[]
                    "Q_shape=$(size(q.Q)), L_shape=$(size(q.L)), K_shape=$(size(q.K)), K_eigenvalues=$(eigvals(q.K))"
                end)
            geometry = Ref{Any}(nothing)
            safe_measure(rows, "B7_hdf5_geometry_enrichment_h11_2",
                () -> begin
                    geometry[] = RD.geometry(idx)
                    geometry[]
                end;
                samples=3,
                summary=() -> begin
                    g = geometry[]
                    "h21=$(g.h21), cy_volume=$(g.cy_volume), glsm_shape=$(size(g.glsm_charges)), divisor_volumes=$(g.τ_volumes)"
                end)
            spectrum = Ref{Any}(nothing)
            safe_measure(rows, "B7_hdf5_pq_query_h11_2",
                () -> begin
                    spectrum[] = G.pq_spectrum(idx; mixing_correction=:float64)
                    spectrum[]
                end;
                samples=2,
                summary=() -> begin
                    s = spectrum[]
                    "m=$(s.m), msign=$(s.msign), f_count=$(length(s.f))"
                end)
        finally
            old_data === nothing ? delete!(ENV, "CYAXIVERSE_DATA_DIR") :
                (ENV["CYAXIVERSE_DATA_DIR"] = old_data)
            old_args === nothing ? delete!(ENV, "newARGS") :
                (ENV["newARGS"] = old_args)
        end
    end
end

function environment_metadata()
    cpu = try
        Sys.cpu_info()[1].model
    catch
        "unavailable"
    end
    julia_build = try
        Base.GIT_VERSION_INFO
    catch
        "unavailable"
    end
    project_file = Base.active_project()
    manifest_file = project_file === nothing ? nothing :
        joinpath(dirname(project_file), "Manifest.toml")
    manifest_sha256 = try
        bytes2hex(open(sha256, manifest_file))
    catch
        "unavailable"
    end
    resolved_versions = try
        dependencies = Pkg.dependencies()
        entries = [begin
            package = info.version === nothing ? "$(info.name)@stdlib" :
                "$(info.name)@$(info.version)"
            "$(uuid):$(package)"
        end for (uuid, info) in dependencies]
        join(sort(entries), ",")
    catch err
        "unavailable:$(sprint(showerror, err))"
    end
    environment_variables = [
        "JULIA_NUM_THREADS=" * get(ENV, "JULIA_NUM_THREADS", "unset"),
        "JULIA_DEPOT_PATH=" * (haskey(ENV, "JULIA_DEPOT_PATH" ) ? "configured" : "unset"),
        "OPENBLAS_NUM_THREADS=" * get(ENV, "OPENBLAS_NUM_THREADS", "unset"),
        "MKL_NUM_THREADS=" * get(ENV, "MKL_NUM_THREADS", "unset"),
        "OMP_NUM_THREADS=" * get(ENV, "OMP_NUM_THREADS", "unset"),
    ]
    compiled_modules = try
        Base.JLOptions().use_compiled_modules
    catch
        "unavailable"
    end
    (; julia_version=VERSION, julia_build, machine=Sys.MACHINE,
       cpu_model=cpu, kernel=Sys.KERNEL, os=Sys.KERNEL,
       arch=Sys.ARCH, blas_vendor=LinearAlgebra.BLAS.vendor(),
       blas_threads=LinearAlgebra.BLAS.get_num_threads(),
       julia_threads=Threads.nthreads(),
       julia_num_threads=get(ENV, "JULIA_NUM_THREADS", "unset"),
       compiled_modules, environment_variables=join(environment_variables, ";"),
       manifest_sha256, resolved_versions,
       active_project="ephemeral_resolved_environment")
end

function main()
    mkpath(OUT)
    rows = NamedTuple[]
    identities = Dict{String,Any}()
    fixtures = direct_fixtures()
    run_b2!(rows, fixtures, identities)
    run_b3!(rows, identities)
    run_b4!(rows, identities)
    run_b5!(rows, fixtures, identities)
    run_b6!(rows, fixtures, identities)
    run_b7!(rows, identities)
    result_path = joinpath(OUT, "b2_b7.tsv")
    metadata_path = joinpath(OUT, "b2_b7_metadata.txt")
    write_rows(result_path, rows)
    execution_revision = try readchomp(`git rev-parse HEAD`) catch; "unavailable" end
    source_branch = try readchomp(`git branch --show-current`) catch; "unavailable" end
    # The reviewed numerical baseline is pinned to 7a40285...; the current
    # branch may carry only contract/evidence documents on top of that source.
    numerical_reference_sha = "7a40285bb5c313f7e8746b90644d5f45bb67be44"
    env = environment_metadata()
    write_metadata(metadata_path;
        generated_utc=Dates.now(Dates.UTC), numerical_reference_sha,
        execution_revision, source_branch,
        repository_relative_results="validation/p0_numerical_equivalence/benchmark_results/b2_b7.tsv",
        method="warmup=1 discarded call; samples=5 by default (route overrides 1--3); GC.gc before every sample; primary allocation=@timed.bytes; crosscheck=@allocated once after warmup; statistic=median with min/max retained",
        environment=env,
        fixture_identities=identities,
        b5_internal_counters="unavailable from current APIs: n8_slow_roll_trajectory does not expose RHS/Hessian counters or final state; correct_stationary_point does not expose Hessian/line-search trial counters",
        b5b_high_residual_tolerance="1e-40 (contract value)",
        b7_data_scope="synthetic bounded HDF5 fixture only; no checked-out geometry data or population scan")
    println("WROTE ", result_path)
    println("WROTE ", metadata_path)
    println("ROWS ", length(rows), " COMPLETED ", count(r -> r.status == "completed", rows), " FAILED ", count(r -> r.status != "completed", rows))
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
