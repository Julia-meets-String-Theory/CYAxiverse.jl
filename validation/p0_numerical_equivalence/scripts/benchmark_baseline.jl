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
       summary=replace(result_summary, '\t' => ' '),
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
    identities["B4_critical_points_n2"] = fixture_digest("B4", q, l, starts)
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
            "starts=$(r.starts), critical_count=$(r.critical_count), minima_count=$(r.minima_count), residuals=$(r.residuals), inertia=$(r.inertia)"
        end)
end

function run_b5!(rows, fixtures, identities)
    # B5a: the bounded current fixed-step N=8 e-fold flow.  The current API
    # exposes accepted step count and exit event but not internal RHS/Hessian
    # counters; those fields remain explicitly unavailable in the report.
    delta_k = 1.0e-3
    identities["B5a_n8_slow_roll"] = fixture_digest("B5a", delta_k, 0.25, 0.1, 1.0e-3, 4)
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
            "entered=$(f.entered_slow_roll), end_event=$(f.end_event), steps=$(f.steps), efolds=$(f.efolds), samples=$(length(f.samples)), solver=$(f.solver)"
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
            "entered=$(p.entered_slow_roll), end_event=$(p.end_event), steps=$(p.steps), efolds=$(p.efolds), samples=$(length(p.samples))"
        end)

    # B5b: stationary correction at Float64 plus a bounded BigFloat replay.
    p = fixtures.p5
    k = Matrix(PB.n5_kinetic_matrix(1.0))
    seed = Float64[0.02, -0.01, 0.03, -0.02, 0.01]
    identities["B5b_stationary_correction_n5"] = fixture_digest("B5b", p.Q, p.L, k, seed)
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
            "status=$(c.status), iterations=$(c.iterations), residual=$(c.residual), internal_seconds=$(c.seconds), working_basis=$(c.working_basis)"
        end)
    comparison = Ref{Any}(nothing)
    safe_measure(rows, "B5b_stationary_correction_float64_bigfloat128_n5",
        () -> begin
            comparison[] = IP.compare_precision(seed, p.Q, p.L, k;
                precision_bits=128, float_residual_tolerance=1e-10,
                high_residual_tolerance=1e-30, zero_tolerance=1e-10,
                max_iterations=40, max_line_search=12)
            comparison[]
        end;
        samples=1,
        summary=() -> begin
            c = comparison[]
            "accepted=$(c.accepted), residual_agreement=$(c.residual_agreement), inertia_agreement=$(c.inertia_agreement), float_status=$(c.float_correction.status), high_status=$(c.high_correction.status), high_residual=$(c.high_correction.residual)"
        end)
end

function run_b6!(rows, fixtures, identities)
    p = fixtures.p5
    k = p isa NamedTuple ? PB.n5_kinetic_matrix(1.0) : nothing
    identities["B6_pq_spectrum_n5"] = fixture_digest("B6", p.Q, p.L, k)
    float_result = Ref{Any}(nothing)
    safe_measure(rows, "B6_pq_spectrum_float64_n5",
        () -> begin
            float_result[] = G.pq_spectrum(k, p.L, p.Q;
                mixing_correction=:float64, mass_basis_diagnostics=true,
                hierarchy_diagnostics=true)
            float_result[]
        end;
        samples=2,
        summary=() -> begin
            s = float_result[]
            "m=$(s.m), msign=$(s.msign), quartic_self_count=$(length(s.λself)), lambda31_count=$(length(s.λ31)), lambda22_count=$(length(s.λ22)), diagnostics=$(s.mass_basis_diagnostics)"
        end)
    high_result = Ref{Any}(nothing)
    safe_measure(rows, "B6_pq_spectrum_high_precision80_n5",
        () -> begin
            high_result[] = G.pq_spectrum(k, p.L, p.Q;
                mixing_correction=:high_precision, prec=80)
            high_result[]
        end;
        samples=1,
        summary=() -> begin
            s = high_result[]
            "m=$(s.m), msign=$(s.msign), quartic_self_count=$(length(s.λself)), lambda31_count=$(length(s.λ31)), lambda22_count=$(length(s.λ22))"
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
        b5_internal_counters="unavailable from current APIs: n8_slow_roll_trajectory does not expose RHS/Hessian counters; correct_stationary_point does not expose Hessian/line-search trial counters",
        b7_data_scope="synthetic bounded HDF5 fixture only; no checked-out geometry data or population scan")
    println("WROTE ", result_path)
    println("WROTE ", metadata_path)
    println("ROWS ", length(rows), " COMPLETED ", count(r -> r.status == "completed", rows), " FAILED ", count(r -> r.status != "completed", rows))
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
