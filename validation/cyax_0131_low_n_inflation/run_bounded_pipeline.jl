#!/usr/bin/env julia

"""Run the bounded CYAX-0131 N=5/N=8 calibration and discovery slice."""

using CYAxiverse
using LinearAlgebra
using SHA
# TOML is bundled with Julia but is not listed in the immutable package project.
const CYAX131_TOML = Base.require(Base.PkgId(
    Base.UUID("fa267f1f-6049-4f14-aa54-33bafae1ed76"), "TOML"))

if !isdefined(Main, :PhaseVolumeDetuningScan)
    include(joinpath(@__DIR__, "..", "..", "scripts",
        "phase_volume_detuning_scan.jl"))
end

const CYAX131 = CYAxiverse.paper_benchmarks
const CYAX131_POLY102 = CYAX131.poly102_inflation
const CYAX131_SCAN = Main.PhaseVolumeDetuningScan
const CYAX131_SCHEMA = "cyax-0131-low-n-inflation-r2"
const CYAX131_N8_PHYSICAL_DELTA_K = 1.5320548620798324e-3
const CYAX131_N8_PHYSICAL_HORIZONS = (1e4, 1e5)
const CYAX131_N8_FOLLOWUP_WALL_BUDGET_SECONDS = 300.0
const CYAX131_N8_PHYSICAL_RELTOL = 1e-8
const CYAX131_N8_PHYSICAL_ABSTOL = 1e-10
const CYAX131_N8_PHYSICAL_MAXITERS = 1_000_000

function _cyax131_stage(message)
    println("[CYAX131] ", message)
    flush(stdout)
    nothing
end

"""Return the fixed, serializable search configuration for this bounded run."""
function cyax131_scan_configuration()
    (; schema_version=CYAX131_SCHEMA,
       benchmark_source="arXiv:2608.14780v1",
       selection_route="paper_named_appendix_fixtures",
       counting_unit="geometry",
       k_homotopy=(; values=[0.5, 0.75, 1.0, 1.25, 1.5],
           meaning="phase_volume_detuning_scan mathematical amplitude homotopy",
           scale_status="homotopy_only"),
       k_physical=(; convention="VOLUME_SCALING_CONVENTION",
           meaning="paper benchmark radial scaling of qdot_tau with inverse-square kinetic metric",
           scale_status="benchmark_convention"),
       n5=(; geometry="poly-102 N5 appendix-B fixture",
           search="zero-phase reduced-model continuation",
           phase_convention="additive argument radians",
           homotopy_coordinate_convention="raw theta radians divided by 2pi; additive phase radians divided by 2pi to cycles",
           k_physical_offsets=["-1e-3", "-2e-4", "+2e-4", "+1e-3"],
           precision_bits=128, gradient_tolerance="1e-30",
           hessian_tolerance="1e-20", event_scale_tolerance="1e-20",
           homotopy_screen="eight-row N5 benchmark Hessian at theta=zeros(5)",
           homotopy_basis="n5_potential(k=1.0); leading eight-row source coefficients",
           k_homotopy_grid=[0.5, 0.75, 1.0, 1.25, 1.5],
           homotopy_phase_vectors="zero phase and instanton-1 delta=0.04 rad probe, converted to cycles",
           phase_probe_source_status="deterministic diagnostic probe; full nonzero vector not supplied by paper",
           phase_probe_comparison="all fixed-theta homotopy eigenvalues retained per grid endpoint in homotopy_attempts.csv",
           homotopy_precision_bits=128, homotopy_tolerance="1e-24"),
       n8=(; geometry="poly-102 N8 appendix-C fixture",
           physical_model="existing ten-row trajectory truncation",
           discovery_initial_condition="n8_degenerate_point seed; theta in period-one coordinates",
           phases="zero phase and explicit single-instanton delta=0.04 rad probe",
           phase_probe_source_status="deterministic diagnostic probe; full nonzero vector not supplied by paper",
           phase_convention_homotopy="raw theta radians and benchmark probe radians divided by 2pi; helper uses cycles",
           k_homotopy_grid=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
           precision_bits=128, homotopy_tolerance="1e-24",
           physical_reestablishment="n8_degenerate_point under paper benchmark physical k",
           physical_trajectory_detuning=string(CYAX131_N8_PHYSICAL_DELTA_K),
           physical_trajectory_precision_bits=100,
           physical_trajectory_time_horizons=collect(CYAX131_N8_PHYSICAL_HORIZONS),
           physical_trajectory_reltol=string(CYAX131_N8_PHYSICAL_RELTOL),
           physical_trajectory_abstol=string(CYAX131_N8_PHYSICAL_ABSTOL),
           physical_trajectory_maxiters=CYAX131_N8_PHYSICAL_MAXITERS,
           physical_trajectory_scan_step=5,
           physical_trajectory_max_step=100,
           physical_trajectory_initial_step=1e-5,
           physical_trajectory_sample_count=20,
           physical_trajectory_followup_wall_budget_seconds=CYAX131_N8_FOLLOWUP_WALL_BUDGET_SECONDS,
           trajectory_observable_sample="last stored sample; exact index recorded"),
       stop_rules=(; phases_deterministic=true,
           no_observational_optimization=true,
           no_population_scan=true,
           no_homotopy_to_physical_transfer=true,
           no_new_as_or_r_convention=true))
end

"""Serialize manifest data with a stable key order."""
function cyax131_manifest_text(manifest)
    io = IOBuffer()
    CYAX131_TOML.print(io, manifest; sorted=true)
    String(take!(io))
end

function _cyax131_csv_escape(value)
    value === nothing && return ""
    value === missing && return ""
    text = value isa AbstractArray ? join(string.(value), ";") : string(value)
    occursin(r"[,\"\n\r]", text) ? string('"', replace(text, '"' => "\"\""), '"') : text
end

function _cyax131_write_attempts(path, runs)
    fields = (:scan_id, :phase_convention, :scale_status, :phase_index,
        :phase, :interval_index, :k_homotopy_low, :k_homotopy_high,
        :eigenvalue_at_k_homotopy_low, :eigenvalue_at_k_homotopy_high,
        :status, :k_homotopy_c, :refined_bracket_width,
        :refined_residual, :refinement_iterations, :precision_bits, :error)
    open(path, "w") do io
        println(io, join(string.(fields), ','))
        for (scan_id, report, phase_convention) in runs
            for attempt in report.attempts
                row = (; scan_id, phase_convention,
                    scale_status=report.scale_status, attempt...)
                println(io, join((_cyax131_csv_escape(getproperty(row, field))
                    for field in fields), ','))
            end
        end
    end
    path
end

function _cyax131_write_trajectory_attempts(path, attempts)
    fields = ("max_time", "wall_seconds", "status", "end_event", "terminated",
        "precision_bits", "reltol", "abstol", "maxiters", "entered_slow_roll",
        "sample_count", "N_e_value", "accepted_steps",
        "rejected_steps", "rhs_evaluations", "jacobian_evaluations",
        "wall_budget_seconds", "error")
    open(path, "w") do io
        println(io, join(fields, ','))
        for attempt in attempts
            println(io, join((_cyax131_csv_escape(get(attempt, field, ""))
                for field in fields), ','))
        end
    end
    path
end

function _cyax131_scan_summary(name, report)
    (; scan_id=name, scale_status=string(report.scale_status),
       phase_count=report.phase_count, k_point_count=report.k_point_count,
       interval_denominator=report.interval_denominator,
       success_count=report.success_count,
       rejected_count=report.rejected_count,
       failure_count=report.failure_count,
       coverage_status=string(report.coverage_status),
       candidates=length(report.candidates))
end

function _cyax131_homotopy_scans(critical_n8)
    n5_source = CYAX131_POLY102.n5_potential(k=1.0)
    n5_probe = CYAX131.phase_fixture(:n5; delta=0.04)
    n5_phases = [zeros(length(n5_source.phases)), n5_probe.phases ./ (2π)]
    k_homotopy_grid_n5 = BigFloat[0.5, 0.75, 1.0, 1.25, 1.5]
    n5_report = CYAX131_SCAN.scan_detailed(zeros(5), Matrix(n5_source.Q'),
        Matrix(n5_source.L'); k_homotopy_grid=k_homotopy_grid_n5,
        phases=n5_phases, precision_bits=128, tolerance=big"1e-24")
    n5_zero_phase_value, n5_probe_value = setprecision(BigFloat, 128) do
        zero_value = CYAX131_SCAN.potential(zeros(5), Matrix(n5_source.Q'),
            Matrix(n5_source.L'), n5_phases[1]; k=1.0, precision_bits=128)
        probe_value = CYAX131_SCAN.potential(zeros(5), Matrix(n5_source.Q'),
            Matrix(n5_source.L'), n5_phases[2]; k=1.0, precision_bits=128)
        zero_value, probe_value
    end
    n5_phase_probe_diagnostic = (; point_identity=
            "theta=zeros(5) period-one coordinates; k_homotopy=1.0",
        model_identity="eight-row N5 leading-term homotopy potential",
        source_status="deterministic single-instanton probe; not a paper-supplied full phase vector",
        phase_probe_assignment=string(n5_probe.assignment),
        phase_convention=string(n5_probe.phase_convention),
        phase_probe_vector_radians=n5_probe.phases,
        probe_phase_cycles=n5_phases[2], input_precision_bits=53,
        diagnostic_precision_bits=128,
        zero_phase_value=n5_zero_phase_value,
        probe_value=n5_probe_value,
        absolute_value_difference=abs(n5_zero_phase_value - n5_probe_value),
        values_differ=n5_zero_phase_value != n5_probe_value,
        comparison_rule="reported fixed-point diagnostic only; no paper-supplied expected value or acceptance tolerance",
        physical_observables="NOT_USED")

    n8_source = CYAX131_POLY102.n8_potential(k=1.0, trajectory=true)
    n8_probe = CYAX131.phase_fixture(:n8; delta=0.04, trajectory=true)
    n8_phases = [zeros(10), n8_probe.phases ./ (2π)]
    n8_report = CYAX131_SCAN.scan_detailed(critical_n8.theta ./ (2π),
        Matrix(n8_source.Q'), Matrix(n8_source.L');
        k_homotopy_grid=BigFloat[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        phases=n8_phases, precision_bits=128, tolerance=big"1e-24")
    (; n5_report, n8_report,
       n5_phase_vectors=n5_phases, n8_phase_vectors=n8_phases,
       n5_k_homotopy_grid=k_homotopy_grid_n5, n8_k_homotopy_grid=BigFloat[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
       n5_probe, n8_probe, n5_phase_probe_diagnostic)
end

function _cyax131_n5_replay()
    bits = 128
    setprecision(BigFloat, bits) do
        kc = CYAX131_POLY102._n5_reduced_zero_phase_critical_scale(BigFloat)
        ks = BigFloat[kc - big"1e-3", kc - big"2e-4",
            kc + big"2e-4", kc + big"1e-3"]
        continuation = CYAX131_POLY102.n5_reduced_zero_phase_continuation(ks;
            seed_theta=BigFloat(π) + big"1e-3",
            gradient_tolerance=big"1e-30", hessian_tolerance=big"1e-20",
            event_scale_tolerance=big"1e-20", max_iterations=64)
        diagnostic = CYAX131.n5_catastrophe_diagnostic(
            k=kc, precision_bits=120, tolerance=big"1e-8")
        phase_probe = CYAX131.phase_fixture(:n5; delta=0.04)
        efold_anchors = [CYAX131_POLY102.n5_hilltop_normal_form_efolds(delta)
            for delta in (1e-7, 6.65e-5)]
        (; kc, continuation, diagnostic, efold_anchors,
           phase_probe, geometry=CYAX131_POLY102.n5_geometry())
    end
end

function _cyax131_n8_replay()
    _cyax131_stage("N8 12-row Table 1 augmented replay start")
    table1_seed = [
        0.0, 0.00499839, 0.99500161, 0.75995156,
        0.75004523, 0.24995477, 0.0, 0.75495317,
    ]
    table1 = nothing
    table1_error = ""
    table1_diagnostic = nothing
    table1_diagnostic_error = ""
    try
        table1 = CYAX131.n8_degenerate_point(table1_seed)
    catch error
        table1_error = sprint(showerror, error)
    end
    _cyax131_stage("N8 12-row Table 1 augmented replay status=" *
        (table1 === nothing ? "failed" : string(table1.converged)))
    if table1 !== nothing && table1.converged
        try
            table1_diagnostic = CYAX131.n8_catastrophe_diagnostic(
                theta=table1.theta, k=table1.k, trajectory=false,
                precision_bits=120, tolerance=1e-8)
        catch error
            table1_diagnostic_error = sprint(showerror, error)
        end
    end

    _cyax131_stage("N8 10-row physical-k re-establishment start")
    critical = CYAX131_POLY102.n8_degenerate_point()
    critical.converged || error("N8 physical-k catastrophe re-establishment failed")
    critical_k_offset = BigFloat(critical.k) - BigFloat(CYAX131_POLY102.N8_KC)
    critical_k_tolerance = BigFloat("1e-11")
    abs(critical_k_offset) <= critical_k_tolerance ||
        error("N8 re-established critical scale differs from the validated trajectory anchor")
    diagnostic = CYAX131.n8_catastrophe_diagnostic(theta=critical.theta,
        k=critical.k, trajectory=true, precision_bits=120, tolerance=1e-8)
    zero_phase = zeros(length(CYAX131_POLY102.N8_TAU_TRAJECTORY))
    probe = CYAX131.phase_fixture(:n8; delta=0.04, trajectory=true)
    zero_derivatives = CYAX131_POLY102.n8_potential_derivatives(
        CYAX131_POLY102.N8_BEST_X, CYAX131_POLY102.N8_KC;
        trajectory=true, phases=zero_phase)
    probe_derivatives = CYAX131_POLY102.n8_potential_derivatives(
        CYAX131_POLY102.N8_BEST_X, CYAX131_POLY102.N8_KC;
        trajectory=true, phases=probe.phases)
    _cyax131_stage("N8 10-row physical-k re-establishment complete k=" *
        string(critical.k) * " residuals=(" * string(critical.gradient_residual) *
        "," * string(critical.null_residual) * ")")

    trajectory = nothing
    trajectory_error = ""
    trajectory_attempts = Dict{String,Any}[]
    for (index, horizon) in enumerate(CYAX131_N8_PHYSICAL_HORIZONS)
        _cyax131_stage("N8 physical trajectory start max_time=" * string(horizon) *
            " arithmetic_bits=100 reltol=" * string(CYAX131_N8_PHYSICAL_RELTOL) *
            " abstol=" * string(CYAX131_N8_PHYSICAL_ABSTOL))
        start_time = time()
        try
            candidate = CYAX131_POLY102.n8_physical_gradient_flow(
                CYAX131_N8_PHYSICAL_DELTA_K; precision_bits=100,
                max_time=horizon, scan_step=5, max_step=100,
                initial_step=1e-5, sample_count=20,
                reltol=CYAX131_N8_PHYSICAL_RELTOL,
                abstol=CYAX131_N8_PHYSICAL_ABSTOL,
                maxiters=CYAX131_N8_PHYSICAL_MAXITERS)
            elapsed = time() - start_time
            push!(trajectory_attempts, Dict{String,Any}(
                "max_time" => horizon,
                "precision_bits" => 100,
                "reltol" => CYAX131_N8_PHYSICAL_RELTOL,
                "abstol" => CYAX131_N8_PHYSICAL_ABSTOL,
                "maxiters" => CYAX131_N8_PHYSICAL_MAXITERS,
                "wall_seconds" => elapsed,
                "status" => candidate.terminated ? "finite_exit" :
                    string(candidate.end_event),
                "end_event" => string(candidate.end_event),
                "terminated" => candidate.terminated,
                "entered_slow_roll" => candidate.entered_slow_roll,
                "sample_count" => length(candidate.samples),
                "N_e_value" => string(candidate.efolds),
                "accepted_steps" => candidate.solver.accepted_steps,
                "rejected_steps" => candidate.solver.rejected_steps,
                "rhs_evaluations" => candidate.solver.rhs_evaluations,
                "jacobian_evaluations" => candidate.solver.jacobian_evaluations))
            trajectory = candidate
            _cyax131_stage("N8 trajectory done max_time=" * string(horizon) *
                " wall_seconds=" * string(elapsed) *
                " event=" * string(candidate.end_event) *
                " terminated=" * string(candidate.terminated) *
                " accepted_steps=" * string(candidate.solver.accepted_steps))
            if candidate.terminated
                break
            end
            if index < length(CYAX131_N8_PHYSICAL_HORIZONS) &&
                    elapsed > CYAX131_N8_FOLLOWUP_WALL_BUDGET_SECONDS
                push!(trajectory_attempts, Dict{String,Any}(
                    "max_time" => CYAX131_N8_PHYSICAL_HORIZONS[index + 1],
                    "status" => "not_run_wall_budget_exceeded",
                    "precision_bits" => 100,
                    "reltol" => CYAX131_N8_PHYSICAL_RELTOL,
                    "abstol" => CYAX131_N8_PHYSICAL_ABSTOL,
                    "maxiters" => CYAX131_N8_PHYSICAL_MAXITERS,
                    "wall_budget_seconds" => CYAX131_N8_FOLLOWUP_WALL_BUDGET_SECONDS))
                _cyax131_stage("N8 1e5 follow-up skipped: 1e4 horizon exceeded " *
                    "the 300-second feasibility budget")
                break
            end
        catch error
            elapsed = time() - start_time
            trajectory_error = sprint(showerror, error)
            push!(trajectory_attempts, Dict{String,Any}(
                "max_time" => horizon,
                "precision_bits" => 100,
                "reltol" => CYAX131_N8_PHYSICAL_RELTOL,
                "abstol" => CYAX131_N8_PHYSICAL_ABSTOL,
                "maxiters" => CYAX131_N8_PHYSICAL_MAXITERS,
                "wall_seconds" => elapsed,
                "status" => "solver_error",
                "error" => trajectory_error))
            _cyax131_stage("N8 trajectory failed max_time=" * string(horizon) *
                " wall_seconds=" * string(elapsed) * " error=" * trajectory_error)
            break
        end
    end

    measured_delta_k = trajectory === nothing ? nothing :
        trajectory.k - BigFloat(critical.k)
    trajectory_binding_tolerance = BigFloat("1e-11")
    if trajectory !== nothing
        abs(measured_delta_k - BigFloat(CYAX131_N8_PHYSICAL_DELTA_K)) <=
            trajectory_binding_tolerance ||
            error("N8 trajectory detuning does not bind to the re-established physical critical point")
    end
    sample_index = trajectory === nothing || isempty(trajectory.samples) ?
        nothing : length(trajectory.samples)
    observables = sample_index === nothing ? nothing :
        CYAX131.trajectory_observables(trajectory; sample_index=sample_index)
    (; table1, table1_seed, table1_error, table1_diagnostic,
       table1_diagnostic_error,
       critical, diagnostic, probe, zero_derivatives, probe_derivatives,
       trajectory, trajectory_error, trajectory_attempts,
       sample_index, observables, critical_k_offset,
       critical_k_tolerance, measured_delta_k, trajectory_binding_tolerance,
       geometry=CYAX131_POLY102.n8_geometry())
end

function _cyax131_manifest(benchmark, config, runs, scans, n5, n8,
        driver_sha256, homotopy_helper_sha256)
    scan_summaries = [_cyax131_scan_summary(id, report) for
        (id, report, _) in runs]
    Dict{String,Any}(
        "schema_version" => CYAX131_SCHEMA,
        "handoff" => Dict("id" => "cyax-0131-low-n-inflation-manager-handoff",
            "revision" => 2,
            "sha256" => "6d76d0e43d005f63a40f10c8e3c7d3b8635e031df15bd8e8633ce7ee6d126b9e",
            "detached_control_desk_state" => "READY_TO_DISPATCH",
            "handoff_review_sha256" => "da89c0ace19bbfcb995d87355d9ca53092ff25f5ab42d87cae6fdf52cdbbd2c9"),
        "execution" => Dict("execution_base_commit" => "74fea608684eae746f25ae45b18512f89b862fb0",
            "scan_driver_sha256" => driver_sha256,
            "homotopy_helper_sha256" => homotopy_helper_sha256,
            "candidate_commit_tree_binding" => "external handback; not self-embedded",
            "julia_version" => string(VERSION),
            "architecture" => string(Sys.ARCH),
            "threads" => Threads.nthreads()),
        "compatibility_impact" => Dict(
            "scope" => "script-local PhaseVolumeDetuningScan helper API",
            "intentional_break" => true,
            "renamed_refinement_function" => "refine_catastrophe -> refine_zero_mode_crossing",
            "renamed_candidate_fields" => "k_low/k_high/k_c/eigenvalue_low/eigenvalue_high/catastrophe_type now state k_homotopy and crossing_type explicitly",
            "reason" => "fixed-configuration eigenvalue crossings do not establish stationarity, branch merger, or catastrophe classification",
            "legacy_scan_k_grid_keyword" => "retained by scan wrapper; returned candidate fields use the scientifically explicit names",
            "package_api_or_project_metadata_changed" => false),
        "source" => Dict("paper_identifier" => benchmark.source.identifier,
            "paper_sha256" => benchmark.source.source_sha256,
            "poly102_source_sha256" => bytes2hex(sha256(read(joinpath(@__DIR__, "..", "..",
                "src", "paper_benchmarks", "poly102_inflation.jl")))),
            "catastrophe_diagnostics_source_sha256" => bytes2hex(sha256(read(joinpath(
                @__DIR__, "..", "..", "src", "paper_benchmarks",
                "catastrophe_diagnostics.jl")))),
            "input_digest" => benchmark.input_digest,
            "selection_route" => string(benchmark.selection_route),
            "counting_unit" => string(benchmark.counting_unit),
            "historical_document_discrepancy" =>
                "validation/inflation_reproduction_results.md retains N5 reduced kc=0.674506370003365; current poly102_inflation.jl and reduced_models.jl source/test route gives N5 kc=1.7700681326109957 (Issue #148 G1 correction, commit 2a4e495ccdd838cc5b1e884fbac136115a7d433f); 0.674506370003365 is N8_KC. Bound document left unchanged."),
        "configuration" => Dict("n5_search" => Dict(
                "method" => config.n5.search,
                "k_physical_offsets_from_kc" => config.n5.k_physical_offsets,
                "precision_bits" => config.n5.precision_bits,
                "gradient_tolerance" => config.n5.gradient_tolerance,
                "hessian_tolerance" => config.n5.hessian_tolerance,
                "event_scale_tolerance" => config.n5.event_scale_tolerance),
            "n5_homotopy" => Dict("k_semantics" => "k_homotopy",
                "scale_status" => "homotopy_only",
                "k_homotopy_grid" => string.(collect(BigFloat.(scans.n5_k_homotopy_grid))),
                "phase_vectors_cycles" => scans.n5_phase_vectors,
                "benchmark_basis" => "n5_potential(k=1.0) Eq. (32) leading eight-row L values",
                "phase_fixture_assignment" => string(scans.n5_probe.assignment),
                "phase_fixture_convention" => string(scans.n5_probe.phase_convention),
                "phase_fixture_vector_radians" => scans.n5_probe.phases,
                "phase_probe_source_status" => scans.n5_phase_probe_diagnostic.source_status,
                "phase_probe_diagnostic_point" => scans.n5_phase_probe_diagnostic.point_identity,
                "phase_probe_diagnostic_model" => scans.n5_phase_probe_diagnostic.model_identity,
                "phase_probe_diagnostic_precision_bits" => scans.n5_phase_probe_diagnostic.diagnostic_precision_bits,
                "phase_probe_zero_phase_value" => string(scans.n5_phase_probe_diagnostic.zero_phase_value),
                "phase_probe_value" => string(scans.n5_phase_probe_diagnostic.probe_value),
                "phase_probe_absolute_value_difference" => string(scans.n5_phase_probe_diagnostic.absolute_value_difference),
                "phase_probe_values_differ" => scans.n5_phase_probe_diagnostic.values_differ,
                "phase_probe_comparison_rule" => scans.n5_phase_probe_diagnostic.comparison_rule,
                "phase_probe_physical_observables" => "NOT_USED",
                "theta_cycles" => zeros(5),
                "coordinate_conversion" => config.n5.homotopy_coordinate_convention,
                "k_homotopy_input_precision_bits" => 53,
                "input_precision_bits" => 53),
            "n8_homotopy" => Dict("k_semantics" => "k_homotopy",
                "scale_status" => "homotopy_only",
                "k_homotopy_grid" => string.(collect(BigFloat.(scans.n8_k_homotopy_grid))),
                "phase_convention" => config.n8.phase_convention_homotopy,
                "phase_probe_source_status" => config.n8.phase_probe_source_status,
                "theta_cycles" => "n8_degenerate_point.theta / 2pi",
                "k_homotopy_input_precision_bits" => 53,
                "phase_vectors_cycles" => [Float64.(n8_phase)
                    for n8_phase in scans.n8_phase_vectors],
                "source_input_precision_bits" => 53,
                "precision_bits" => config.n8.precision_bits,
                "tolerance" => config.n8.homotopy_tolerance)),
        "bounded_scans" => [Dict(string(k) => getproperty(summary, k)
            for k in propertynames(summary)) for summary in scan_summaries],
        "n5_physical_reduced_model" => Dict(
            "status" => "completed",
            "geometry_identity" => "poly-102 N5 appendix-B named fixture",
            "h11" => n5.geometry.h11,
            "h21" => n5.geometry.h21,
            "euler" => n5.geometry.euler,
            "critical_scale" => string(n5.kc),
            "branch" => "pi",
            "continuation_steps" => length(n5.continuation),
            "converged_steps" => count(step -> step.converged, n5.continuation),
            "catastrophe_events" => count(step -> step.catastrophe_detected,
                n5.continuation),
            "continued_critical_points" => [Dict(
                "k_physical" => string(step.k),
                "delta_k_physical_from_kc" => string(step.k - n5.kc),
                "theta_radians" => string(step.theta),
                "branch_identity" => string(step.branch),
                "gradient_residual" => string(step.gradient),
                "converged" => step.converged,
                "catastrophe_detected" => step.catastrophe_detected)
                for step in n5.continuation],
            "diagnostic_classification" => string(n5.diagnostic.classification),
            "diagnostic_precision_bits" => n5.diagnostic.precision_bits,
            "phase_probe_assignment" => string(n5.phase_probe.assignment),
            "phase_probe_convention" => string(n5.phase_probe.phase_convention),
            "phase_probe_source_status" => "deterministic diagnostic probe; not a paper-supplied full phase vector",
            "phase_probe_vector_radians" => n5.phase_probe.phases,
            "reduced_efold_anchors" => [Dict("delta_k" => string(anchor.delta_k),
                "N_e" => string(anchor.efolds)) for anchor in n5.efold_anchors],
            "full_physical_N5_trajectory_observables" => "NOT_REACHED"),
        "n8_twelve_row_table1_replay" => Dict(
            "model_identity" => "paper Table 1 source; twelve instanton rows",
            "route" => "paper_benchmarks.n8_degenerate_point augmented solve",
            "input_seed_period_one" => n8.table1_seed,
            "status" => n8.table1 === nothing ? "failed" :
                n8.table1.converged ? "converged" : "not_converged",
            "k_physical" => n8.table1 === nothing ? "NOT_REACHED" :
                string(n8.table1.k),
            "gradient_residual" => n8.table1 === nothing ? "NOT_REACHED" :
                string(n8.table1.gradient_residual),
            "null_residual" => n8.table1 === nothing ? "NOT_REACHED" :
                string(n8.table1.null_residual),
            "iterations" => n8.table1 === nothing ? 0 : n8.table1.iterations,
            "error" => n8.table1_error,
            "catastrophe_diagnostic_status" => n8.table1_diagnostic === nothing ?
                "NOT_REACHED" : string(n8.table1_diagnostic.classification),
            "catastrophe_diagnostic_point_identity" =>
                "12-row Table 1 augmented replay returned theta and k; diagnostic uses twelve-row source model",
            "catastrophe_diagnostic_precision_bits" => n8.table1_diagnostic === nothing ?
                120 : n8.table1_diagnostic.precision_bits,
            "catastrophe_diagnostic_tolerance" => "1e-8",
            "catastrophe_diagnostic_error" => n8.table1_diagnostic_error,
            "physical_trajectory_authorized" => false),
        "n8_physical_reestablishment" => Dict(
            "status" => "completed",
            "model" => "existing ten-row poly-102 trajectory truncation",
            "geometry_identity" => "poly-102 N8 appendix-C named fixture",
            "h11" => n8.geometry.h11,
            "h21" => n8.geometry.h21,
            "euler" => n8.geometry.euler,
            "route" => "n8_degenerate_point; zero phase; paper benchmark physical k",
            "k_physical" => string(n8.critical.k),
            "gradient_residual" => string(n8.critical.gradient_residual),
            "null_residual" => string(n8.critical.null_residual),
            "input_precision_bits" => 53,
            "critical_point_precision_note" => "n8_degenerate_point and its k are Float64 (53-bit input) at the source-defined 1e-11 tolerance; the 120-bit diagnostic and 100-bit trajectory do not refine this critical scale",
            "solver_tolerance" => string(n8.critical.tolerance),
            "critical_scale_offset_from_trajectory_anchor" =>
                string(n8.critical_k_offset),
            "critical_scale_binding_tolerance" => string(n8.critical_k_tolerance),
            "iterations" => n8.critical.iterations,
            "homotopy_candidates_promoted" => 0),
        "n8_catastrophe_point_diagnostic" => Dict(
            "point_identity" => "n8_degenerate_point zero-phase ten-row theta and k",
            "classification" => string(n8.diagnostic.classification),
            "normal_form" => string(n8.diagnostic.normal_form),
            "precision_bits" => n8.diagnostic.precision_bits,
            "point_input_precision_bits" => 53,
            "tolerance" => "1e-8",
            "transverse_hessian_eigenvalues" => string.(
                n8.diagnostic.transverse_hessian_eigenvalues)),
        "n8_phase_probe_diagnostic" => Dict(
            "point_identity" => "N8_BEST_X; ten-row trajectory source; physical benchmark k=kc; diagnostic evaluation only",
            "probe_kind" => "deterministic single-instanton 0.04-radian diagnostic probe",
            "probe_source_status" => "not a paper-supplied complete nonzero phase vector",
            "phase_probe_assignment" => string(n8.probe.assignment),
            "phase_convention" => string(n8.probe.phase_convention),
            "phase_probe_vector_radians" => n8.probe.phases,
            "zero_phase_value" => string(n8.zero_derivatives.value),
            "single_instanton_probe_value" => string(n8.probe_derivatives.value),
            "absolute_value_difference" => string(abs(
                n8.zero_derivatives.value - n8.probe_derivatives.value)),
            "values_differ" => n8.zero_derivatives.value != n8.probe_derivatives.value,
            "input_precision_bits" => 53,
            "comparison_tolerance" => "0.0; exact Float64 value inequality only, with no source-matched expected value",
            "physical_observables_from_probe" => "NOT_USED"),
        "n8_physical_trajectory" => Dict(
            "status" => n8.trajectory === nothing ? "solver_error" :
                n8.trajectory.terminated ? "finite_exit" :
                n8.trajectory.end_event === :no_slow_roll_window ?
                    "no_slow_roll_window_at_horizon" : "censored_at_tmax",
            "trajectory_route" => "existing n8_physical_gradient_flow / n8_author_trajectory",
            "full_physical_benchmark_replay" =>
                n8.trajectory !== nothing && n8.trajectory.terminated ?
                    "OBSERVED_FINITE_EXIT" : "NOT_VERIFIED",
            "error" => n8.trajectory_error,
            "k_physical" => n8.trajectory === nothing ? "NOT_REACHED" :
                string(n8.trajectory.k),
            "requested_delta_k_physical_from_anchor" =>
                string(CYAX131_N8_PHYSICAL_DELTA_K),
            "measured_delta_k_physical_from_reestablished_kc" => n8.measured_delta_k === nothing ?
                "NOT_REACHED" :
                string(n8.measured_delta_k),
            "trajectory_detuning_binding_residual" => n8.measured_delta_k === nothing ?
                "NOT_REACHED" : string(
                    n8.measured_delta_k - BigFloat(CYAX131_N8_PHYSICAL_DELTA_K)),
            "trajectory_detuning_binding_tolerance" =>
                string(n8.trajectory_binding_tolerance),
            "requested_delta_input_precision_bits" => 53,
            "precision_bits" => n8.trajectory === nothing ? 100 :
                n8.trajectory.precision_bits,
            "arithmetic_precision_claim" => "100-bit BigFloat arithmetic; explicit 1e-8 relative and 1e-10 absolute solver error tolerances do not imply 100-bit observable accuracy",
            "integration_reltol" => string(CYAX131_N8_PHYSICAL_RELTOL),
            "integration_abstol" => string(CYAX131_N8_PHYSICAL_ABSTOL),
            "maxiters" => CYAX131_N8_PHYSICAL_MAXITERS,
            "scan_step" => 5,
            "max_step" => 100,
            "initial_step" => 1e-5,
            "sample_count" => 20,
            "N_e" => n8.trajectory !== nothing && n8.trajectory.terminated ?
                string(n8.trajectory.efolds) : "NOT_REACHED",
            "N_e_lower_bound_at_tmax" => n8.trajectory !== nothing &&
                n8.trajectory.end_event === :tmax ?
                    string(n8.trajectory.efolds) : "NOT_APPLICABLE",
            "N_e_status" => n8.trajectory === nothing ? "NOT_REACHED" :
                n8.trajectory.terminated ? "finite_exit" :
                n8.trajectory.end_event === :tmax ? "lower_bound_censored_at_tmax" :
                "NOT_REACHED_no_slow_roll_window",
            "slow_roll_window_efolds" => n8.trajectory === nothing ?
                "NOT_REACHED" : string(n8.trajectory.slow_roll_efolds),
            "end_event" => n8.trajectory === nothing ? "solver_error" :
                string(n8.trajectory.end_event),
            "terminated" => n8.trajectory !== nothing && n8.trajectory.terminated,
            "n_s" => n8.observables === nothing ? "NOT_REACHED" :
                string(n8.observables.n_s),
            "paper_delta_H" => n8.observables === nothing ? "NOT_REACHED" :
                string(n8.observables.delta_H),
            "scalar_amplitude_convention" => n8.observables === nothing ?
                "NOT_REACHED" : string(n8.observables.scalar_amplitude_convention),
            "cumulative_turning" => n8.observables === nothing ? "NOT_REACHED" :
                string(n8.observables.cumulative_turning),
            "sample_index" => n8.sample_index === nothing ? 0 : n8.sample_index,
            "sample_index_convention" => "1-based exact stored sample index; 0 means no sample",
            "sample_interpretation" => "last stored sample at the observed finite exit; diagnostic only, not an observational pivot or viability test",
            "sample_coordinate_n" => n8.sample_index === nothing ? "NOT_REACHED" :
                string(n8.trajectory.samples[n8.sample_index].n),
            "trajectory_attempts" => n8.trajectory_attempts,
            "n8_ne_e_folds_validated_tight_step_reference" => "approximately 60.00336 (bound validation/inflation_reproduction_results.md; tight-step convergence, not an exact acceptance target)",
            "n8_ne_e_folds_tight_step_difference" => n8.trajectory !== nothing &&
                n8.trajectory.terminated ? string(n8.trajectory.efolds -
                    BigFloat("60.00336")) : "NOT_REACHED",
            "n8_ne_e_folds_acceptance_tolerance" => "none defined for this approximate tight-step reference; difference reported without acceptance",
            "n8_coarse_float64_bdf_efolds_provenance" => "59.690642055250756; coarse Float64 BDF max_step=100 value retained as provenance only, not the governing physical-flow reference",
            "n8_coarse_float64_bdf_difference" => n8.trajectory !== nothing &&
                n8.trajectory.terminated ? string(n8.trajectory.efolds -
                    BigFloat("59.690642055250756")) : "NOT_REACHED",
            "A_s_conversion_and_acceptance_windows" => "NOT_REACHED",
            "n_s_acceptance_window" => "NOT_REACHED",
            "tensor_to_scalar_ratio_r" => "NOT_REACHED"),
        "claim_boundary" => Dict("fixed_saxions" => true,
            "observational_viability" => "NOT_CLAIMED",
            "issue_131_observational_stretch_goal" => "NOT_REACHED_IN_R2",
            "population_scan" => "NOT_REACHED",
            "dynamical_saxion_stabilisation" => "NOT_REACHED",
            "merge" => "NOT_AUTHORIZED",
            "issue_131_closure" => "NOT_AUTHORIZED"))
end

"""Run and write the finite r2 evidence packet beneath the approved subtree."""
function run_cyax131_bounded_pipeline(output_dir::AbstractString=@__DIR__)
    mkpath(output_dir)
    _cyax131_stage("bounded pipeline start schema=" * CYAX131_SCHEMA)
    benchmark = CYAX131.benchmark_manifest()
    config = cyax131_scan_configuration()
    _cyax131_stage("N5 reduced-model replay start")
    n5 = _cyax131_n5_replay()
    _cyax131_stage("N5 reduced-model replay complete kc=" * string(n5.kc) *
        " continuation_points=" * string(length(n5.continuation)))
    _cyax131_stage("N8 benchmark and trajectory replay start")
    n8 = _cyax131_n8_replay()
    _cyax131_stage("N8 benchmark and trajectory replay complete status=" *
        (n8.trajectory === nothing ? "solver_error" :
            n8.trajectory.terminated ? "finite_exit" : string(n8.trajectory.end_event)))
    _cyax131_stage("bounded homotopy scans start")
    scans = _cyax131_homotopy_scans(n8.critical)
    _cyax131_stage("bounded homotopy scans complete")
    runs = [("n5_poly102_homotopy_screen", scans.n5_report,
             "cycles; zero phase and single-instanton probe"),
            ("n8_poly102_homotopy_screen", scans.n8_report,
             config.n8.phase_convention_homotopy)]
    driver_path = @__FILE__
    driver_sha256 = bytes2hex(sha256(read(driver_path)))
    helper_path = joinpath(@__DIR__, "..", "..", "scripts",
        "phase_volume_detuning_scan.jl")
    homotopy_helper_sha256 = bytes2hex(sha256(read(helper_path)))
    manifest = _cyax131_manifest(benchmark, config, runs, scans, n5, n8,
        driver_sha256, homotopy_helper_sha256)
    open(joinpath(output_dir, "scan_manifest.toml"), "w") do io
        write(io, cyax131_manifest_text(manifest))
    end
    _cyax131_write_attempts(joinpath(output_dir, "homotopy_attempts.csv"), runs)
    _cyax131_write_trajectory_attempts(
        joinpath(output_dir, "n8_trajectory_attempts.csv"), n8.trajectory_attempts)
    _cyax131_stage("bounded pipeline outputs written to " * output_dir)
    manifest
end

if abspath(PROGRAM_FILE) == @__FILE__
    output_dir = isempty(ARGS) ? (@__DIR__) : ARGS[1]
    manifest = run_cyax131_bounded_pipeline(output_dir)
    println("CYAX-0131 bounded run wrote ", output_dir)
    println("N5 reduced continuation: ", manifest["n5_physical_reduced_model"])
    println("N8 physical trajectory: ", manifest["n8_physical_trajectory"])
    println("homotopy scans: ", manifest["bounded_scans"])
end
