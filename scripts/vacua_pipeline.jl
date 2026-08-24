using CYAxiverse
using HDF5
using LinearAlgebra
using Printf
using Dates
using SHA

# This file is included by both the test harness and the batch runner.  Keep
# the helper definitions behind a module-local guard so a second include is a
# no-op while direct invocation still reaches the CLI below.
if !isdefined(@__MODULE__, :VACUA_PIPELINE_VERSION)

const VACUA_PIPELINE_VERSION = "1"
const VACUA_CLASSIFICATION_SCHEMA_VERSION = "cyaxiverse-vacua-classification-2"
const VACUA_CLASSIFICATION_LEGACY_ALIASES = Dict(
    "exact_determinant_branch" => "square_reduced_potential_determinant_count",
    "certified_selected_branch_set" => "complete_selected_leading_branch_count",
    "finite_search_lower_bound" => "finite_multistart_minimum_lower_bound",
)

"""Return the approved public classification for a legacy or current label."""
function _canonical_vacua_classification(label)
    value = string(label)
    get(VACUA_CLASSIFICATION_LEGACY_ALIASES, value, value)
end

"""Return the pre-schema-2 alias for a current classification, when known."""
function _legacy_vacua_classification(label)
    value = string(label)
    for (legacy, current) in VACUA_CLASSIFICATION_LEGACY_ALIASES
        current == value && return legacy
    end
    value
end

"""Normalize persisted vacuum-search metadata and record legacy aliases."""
function _normalize_vacua_search_metadata(search_metadata)
    search_metadata === nothing && return nothing
    hasproperty(search_metadata, :search_classification) || return search_metadata
    classification = _canonical_vacua_classification(
        getproperty(search_metadata, :search_classification))
    merge(search_metadata, (; search_classification=classification,
        legacy_search_classification=_legacy_vacua_classification(classification),
        classification_schema_version=VACUA_CLASSIFICATION_SCHEMA_VERSION,
        model_scope="hierarchy_truncated_axion_potential",
        full_potential_status="not_validated"))
end

"""Return the short Git revision for provenance metadata."""
function _git_revision()
    root = dirname(@__DIR__)
    try
        readchomp(`git -C $root rev-parse --short HEAD`)
    catch
        "unknown"
    end
end

"""Validate and normalize an explicitly supplied geometry data directory."""
function _validate_data_dir(data_dir::AbstractString)
    isempty(strip(data_dir)) && throw(ArgumentError("data_dir must be explicitly provided"))
    path = abspath(expanduser(data_dir))
    path in ("/", homedir()) &&
        throw(ArgumentError("refusing to use root-like data directory: $path"))
    isdir(path) || throw(ArgumentError("data directory does not exist: $path"))
    path
end

"""Validate the dimensions and finite values of one geometry potential."""
function _validate_potential(geom_idx, pot_data)
    h11 = geom_idx.h11
    Q, L, K = pot_data.Q, pot_data.L, pot_data.K
    size(Q, 1) == h11 || throw(DimensionMismatch("Q must have h11 rows"))
    size(Q, 2) > h11 || throw(DimensionMismatch("Q must contain more instantons than axions"))
    size(L) == (2, size(Q, 2)) ||
        throw(DimensionMismatch("L must have shape (2, size(Q, 2))"))
    size(K) == (h11, h11) || throw(DimensionMismatch("K must have shape (h11, h11)"))
    all(isfinite, L) || throw(ArgumentError("L contains non-finite values"))
    all(isfinite, Matrix(K)) || throw(ArgumentError("K contains non-finite values"))
    nothing
end

"""Build the configuration tuple persisted with a vacua pipeline result."""
function _pipeline_config(; threshold, starts, residual_tolerance, merge_tolerance,
    max_iterations, method, max_branches::Int=1_000_000)
    (; pipeline_version=VACUA_PIPELINE_VERSION, threshold, starts,
       residual_tolerance, merge_tolerance, max_iterations, method=string(method),
       max_branches, branch_method=string(method))
end

"""Hash a persisted vacua configuration for stale-result detection."""
_pipeline_config_digest(config) = bytes2hex(sha256(repr(config)))

"""Return whether a completed result matches an optional saved configuration."""
function _has_pipeline_result(path; config=nothing)
    isfile(path) || return false
    h5open(path, "r") do file
        haskey(file, "vacua_pipeline/metadata/status") || return false
        read(file["vacua_pipeline/metadata/status"]) == "completed" || return false
        haskey(file, "vacua_pipeline/metadata/terminal_status") || return false
        read(file["vacua_pipeline/metadata/terminal_status"]) == "completed" || return false
        config === nothing && return true
        metadata = file["vacua_pipeline/metadata"]
        haskey(metadata, "configuration_digest") || return false
        read(metadata["configuration_digest"]) == _pipeline_config_digest(config) || return false
        required = (("pipeline_version", config.pipeline_version),
            ("threshold", config.threshold), ("starts", config.starts),
            ("residual_tolerance", config.residual_tolerance),
            ("merge_tolerance", config.merge_tolerance),
            ("max_iterations", config.max_iterations), ("method", config.method),
            ("max_branches", config.max_branches))
        all(name -> haskey(metadata, name[1]) && read(metadata[name[1]]) == name[2], required)
    end
end

"""Return whether a geometry file contains any persisted vacua group."""
function _pipeline_group_exists(path)
    isfile(path) || return false
    h5open(path, "r") do file
        haskey(file, "vacua_pipeline")
    end
end

"""Return whether an HDF5 geometry file contains a vacua pipeline group."""
function _has_pipeline_group(path; config=nothing)
    isfile(path) || return false
    h5open(path, "r") do file
        haskey(file, "vacua_pipeline") || return false
        haskey(file, "vacua_pipeline/metadata/status") || return false
        read(file["vacua_pipeline/metadata/status"]) == "completed" || return false
        config === nothing ? true : _has_pipeline_result(path; config)
    end
end

"""Write solver, provenance, status, and search metadata to an HDF5 group."""
function _write_metadata!(group, config; status, estimate_status, verification_status,
    runtime_seconds, solver_status=status, search_metadata=nothing, error_message="",
        timings=nothing)
    metadata = create_group(group, "metadata")
    for (name, value) in pairs(config)
        metadata[string(name)] = value
    end
    metadata["status"] = status
    metadata["terminal_status"] = status
    metadata["configuration_digest"] = _pipeline_config_digest(config)
    metadata["solver_status"] = solver_status
    metadata["estimate_status"] = estimate_status
    metadata["verification_status"] = verification_status
    metadata["julia_version"] = string(VERSION)
    metadata["git_revision"] = _git_revision()
    metadata["runtime_seconds"] = runtime_seconds
    metadata["completed_at"] = string(Dates.now())
    metadata["error"] = error_message
    normalized_search_metadata = _normalize_vacua_search_metadata(search_metadata)
    if normalized_search_metadata !== nothing
        for (name, value) in pairs(normalized_search_metadata)
            metadata[string(name)] = value
        end
    end
    _write_timings!(metadata, timings)
end

"""Record a monotonic wall-clock duration in a string-keyed timing map."""
function _record_stage!(timings::AbstractDict{String, <:Real}, name::AbstractString,
        started_ns::UInt64)
    timings[string(name)] = (time_ns() - started_ns) / 1e9
    timings[string(name)]
end

"""Persist optional stage timings without changing scientific datasets."""
function _write_timings!(metadata, timings)
    timings === nothing && return
    for (name, seconds) in pairs(timings)
        metadata[string("timing_", name)] = seconds
    end
end

"""Run the legacy finite-start search and retain its lower-bound label."""
function _legacy_vacua_search(geom_idx, Q, L; threshold, starts)
    estimate = CYAxiverse.generate.vacua_estimate(geom_idx; threshold=threshold)
    locations = CYAxiverse.generate.vacua_id(L, Q; threshold=threshold, runs=starts)
    search = (; search_method="legacy",
        search_classification="finite_multistart_minimum_lower_bound",
        legacy_search_classification="finite_search_lower_bound",
        classification_schema_version=VACUA_CLASSIFICATION_SCHEMA_VERSION,
        model_scope="hierarchy_truncated_axion_potential",
        full_potential_status="not_validated",
        minimum_count=locations["vac"], multiplicity=1.0,
        critical_count=-1, branch_count=-1, det_Qtilde=-1,
        search_status="completed")
    estimate, locations, search
end

"""Run the deterministic leading selected-branch enumeration."""
function _leading_branch_search(Q, L; max_branches)
    selected = CYAxiverse.generate.LQtilde(Q, L)
    branches = CYAxiverse.generate.leading_critical_branches(selected;
        max_branches=max_branches)
    estimate = if size(selected.Qtilde, 1) == size(selected.Qtilde, 2)
        (; vac=branches.leading_minima_count, issquare=1)
    else
        (; vac=branches.leading_minima_count, issquare=0,
            extrarows=size(selected.Qtilde, 2) - size(selected.Qtilde, 1))
    end
    locations = Dict{String, Int}("vac" => branches.leading_minima_count)
    search = (; search_method="leading_branches",
        search_classification="complete_selected_leading_branch_count",
        legacy_search_classification="certified_selected_branch_set",
        classification_schema_version=VACUA_CLASSIFICATION_SCHEMA_VERSION,
        model_scope="hierarchy_truncated_axion_potential",
        full_potential_status="not_validated",
        minimum_count=branches.leading_minima_count, multiplicity=1.0,
        critical_count=branches.branch_count, branch_count=branches.branch_count,
        det_Qtilde=branches.det_Qtilde, search_status="completed")
    estimate, locations, search
end

"""Solve one already-prepared reduced JLM problem."""
function _reduced_problem_search(problem; starts, residual_tolerance,
        merge_tolerance, max_iterations)
    minimum = CYAxiverse.jlm_reduced.minimize(problem; starts=starts,
        residual_tolerance=residual_tolerance, merge_tolerance=merge_tolerance,
        max_iterations=max_iterations)
    square = minimum isa CYAxiverse.structs.Min_JLM_Square
    estimate = square ? (; vac=minimum.N_min, issquare=1) :
        (; vac=minimum.N_min, issquare=0, extrarows=minimum.extra_rows)
    locations = Dict{String, Int}("vac" => minimum.N_min)
    search = (; search_method="reduced_jlm",
        search_classification=square ? "square_reduced_potential_determinant_count" :
            "finite_multistart_minimum_lower_bound",
        legacy_search_classification=square ? "exact_determinant_branch" :
            "finite_search_lower_bound",
        classification_schema_version=VACUA_CLASSIFICATION_SCHEMA_VERSION,
        model_scope="hierarchy_truncated_axion_potential",
        full_potential_status="not_validated",
        minimum_count=minimum.N_min, multiplicity=problem.multiplicity,
        critical_count=-1, branch_count=-1, det_Qtilde=minimum.det_QTilde,
        search_status="completed")
    estimate, locations, search
end

"""Dispatch one geometry to the selected vacua search method.

`:auto` follows the bounded decision tree exact determinant → selected branch
set → reduced JLM finite search → legacy finite search. Each fallback is
recorded in the returned operational metadata; existing explicit methods are
unchanged.
"""
function _search_vacua(geom_idx, pot_data; threshold, starts, residual_tolerance,
        merge_tolerance, max_iterations, method, max_branches)
    Q, L = pot_data.Q, pot_data.L
    if method == :legacy
        return _legacy_vacua_search(geom_idx, Q, L; threshold, starts)
    elseif method == :leading_branches
        return _leading_branch_search(Q, L; max_branches)
    elseif method == :reduced_jlm
        problem = CYAxiverse.jlm_reduced.prepare(Q, L; threshold=threshold)
        return _reduced_problem_search(problem; starts, residual_tolerance,
            merge_tolerance, max_iterations)
    elseif method == :auto
        problem = CYAxiverse.jlm_reduced.prepare(Q, L; threshold=threshold)
        if problem.square_vacua !== nothing
            result = _reduced_problem_search(problem; starts, residual_tolerance,
                merge_tolerance, max_iterations)
            return result[1], result[2], merge(result[3],
                (; auto_selected_method="exact_determinant"))
        end

        try
            result = _leading_branch_search(Q, L; max_branches)
            return result[1], result[2], merge(result[3],
                (; auto_selected_method="selected_branch_set"))
        catch err
            is_branches_limit = err isa ArgumentError &&
                occursin("leading branch enumeration", sprint(showerror, err))
            is_branches_limit || rethrow()
        end

        try
            result = _reduced_problem_search(problem; starts, residual_tolerance,
                merge_tolerance, max_iterations)
            return result[1], result[2], merge(result[3],
                (; auto_selected_method="reduced_jlm"))
        catch err
            result = _legacy_vacua_search(geom_idx, Q, L; threshold, starts)
            return result[1], result[2], merge(result[3],
                (; auto_selected_method="legacy",
                    auto_fallback_reason=sprint(showerror, err)))
        end
    end
    throw(ArgumentError("unsupported method: $method"))
end

"""
    save_axion_data(geom_idx, spectrum, vac_est, vac_id; kwargs...)

Write vacua results into a temporary copy and atomically replace the target.
The existing spectrum groups in the geometry file are preserved.
"""
function save_axion_data(geom_idx, spectrum, vac_est, vac_id; threshold::Float64,
    starts::Int=10_000, residual_tolerance::Float64=1e-10,
        merge_tolerance::Float64=1e-7, max_iterations::Int=200,
    method::Symbol=:legacy, max_branches::Int=1_000_000,
    search_metadata=nothing, runtime_seconds::Float64=0.0, timings=nothing,
        force::Bool=false)
    target = CYAxiverse.filestructure.cyax_file(geom_idx)
    isfile(target) || throw(ArgumentError("geometry file does not exist: $target"))
    config = _pipeline_config(; threshold, starts, residual_tolerance,
        merge_tolerance, max_iterations, method, max_branches)
    !_pipeline_group_exists(target) || force ||
        throw(ArgumentError("vacua_pipeline already exists; pass force=true to replace $target"))

    temporary = string(target, ".vacua-pipeline.tmp-", getpid(), "-", time_ns())
    cp(target, temporary; force=true)
    try
        h5open(temporary, "r+") do file
            haskey(file, "vacua_pipeline") && HDF5.delete_object(file, "vacua_pipeline")
            vacua_group = create_group(file, "vacua_pipeline")
            vacua_group["threshold", deflate=9] = threshold
            vacua_group["estimate", deflate=9] = vac_est.vac
            vacua_group["issquare", deflate=9] = vac_est.issquare
            if hasproperty(vac_est, :extrarows)
                vacua_group["extrarows", deflate=9] = vac_est.extrarows
            end
            verification_status = if search_metadata === nothing
                haskey(vac_id, "vac") ? "verified" : "not_applicable"
            elseif search_metadata.search_classification in
                    ("square_reduced_potential_determinant_count",
                     "exact_determinant_branch")
                "verified"
            elseif search_metadata.search_classification in
                    ("complete_selected_leading_branch_count",
                     "certified_selected_branch_set")
                "verified_selected_branch_set"
            else
                "not_applicable"
            end
            if haskey(vac_id, "vac")
                vacua_group["verified", deflate=9] = vac_id["vac"]
            end
            for (key, path) in (("θ̃min", "theta_min"), ("θ̃∥", "theta_parallel"))
                if haskey(vac_id, key)
                    coordinates = vac_id[key]
                    coordinates_group = create_group(vacua_group, path)
                    coordinates_group["numerator", deflate=9] = Int.(numerator.(coordinates))
                    coordinates_group["denominator", deflate=9] = Int.(denominator.(coordinates))
                end
            end
            _write_metadata!(vacua_group, config; status="completed",
                estimate_status="estimated", verification_status=verification_status,
                runtime_seconds,
                solver_status=search_metadata === nothing ? "completed" :
                    search_metadata.search_status, search_metadata=search_metadata,
                    timings=timings)
        end
        mv(temporary, target; force=true)
    catch
        isfile(temporary) && rm(temporary; force=true)
        rethrow()
    end
    nothing
end

"""Validate inputs, load one potential, and run only the vacua search."""
function _vacua_core(geom_idx, data_dir::AbstractString; threshold::Float64=0.5,
        starts::Int=10_000, residual_tolerance::Float64=1e-10,
        merge_tolerance::Float64=1e-7, max_iterations::Int=200,
        method::Symbol=:legacy, max_branches::Int=1_000_000)
    started_ns = time_ns()
    ENV["CYAXIVERSE_DATA_DIR"] = _validate_data_dir(data_dir)
    threshold >= 0 || throw(ArgumentError("threshold must be nonnegative"))
    starts > 0 || throw(ArgumentError("starts must be positive"))
    residual_tolerance > 0 || throw(ArgumentError("residual_tolerance must be positive"))
    merge_tolerance > 0 || throw(ArgumentError("merge_tolerance must be positive"))
    max_iterations > 0 || throw(ArgumentError("max_iterations must be positive"))
    method in (:auto, :legacy, :leading_branches, :reduced_jlm) ||
        throw(ArgumentError("unsupported method: $method"))
    max_branches > 0 || throw(ArgumentError("max_branches must be positive"))

    timings = Dict{String, Float64}()
    potential_started_ns = time_ns()
    pot_data = CYAxiverse.read.potential(geom_idx)
    _validate_potential(geom_idx, pot_data)
    _record_stage!(timings, "potential_load_seconds", potential_started_ns)

    search_started_ns = time_ns()
    vac_est, vac_id, search = _search_vacua(geom_idx, pot_data; threshold, starts,
        residual_tolerance, merge_tolerance, max_iterations, method, max_branches)
    _record_stage!(timings, "vacua_search_seconds", search_started_ns)
    timings["total_seconds"] = (time_ns() - started_ns) / 1e9
    (; geom_idx, pot_data, vac_est, vac_id, search, timings, started_ns)
end

"""
    compute_vacua_data(h11, np, cy, data_dir; kwargs...)

Run only the potential/vacua portion of the pipeline. This avoids loading
geometry and computing the full PQ spectrum when a scan needs vacua counts
only. Existing scientific counts, thresholds, and verification labels are
unchanged. Set `save=false` for a read-only computation.
"""
function compute_vacua_data(h11::Int, np::Int, cy::Int, data_dir::AbstractString;
        threshold::Float64=0.5, save::Bool=true, force::Bool=false,
        starts::Int=10_000, residual_tolerance::Float64=1e-10,
        merge_tolerance::Float64=1e-7, max_iterations::Int=200,
        method::Symbol=:legacy, max_branches::Int=1_000_000)
    geom_idx = CYAxiverse.structs.GeometryIndex(h11, np, cy)
    compute_vacua_data(geom_idx, data_dir; threshold, save, force, starts,
        residual_tolerance, merge_tolerance, max_iterations, method, max_branches)
end

"""Run the vacuum-only pipeline for an indexed geometry."""
function compute_vacua_data(geom_idx::CYAxiverse.structs.GeometryIndex,
        data_dir::AbstractString; threshold::Float64=0.5, save::Bool=true,
        force::Bool=false, starts::Int=10_000,
        residual_tolerance::Float64=1e-10, merge_tolerance::Float64=1e-7,
        max_iterations::Int=200, method::Symbol=:legacy,
        max_branches::Int=1_000_000)
    core = _vacua_core(geom_idx, data_dir; threshold, starts, residual_tolerance,
        merge_tolerance, max_iterations, method, max_branches)
    if save
        save_axion_data(geom_idx, nothing, core.vac_est, core.vac_id;
            threshold, starts, residual_tolerance, merge_tolerance, max_iterations,
            method, max_branches, search_metadata=core.search,
            runtime_seconds=core.timings["total_seconds"], timings=core.timings, force)
    end
    return Dict(
        "geom_idx" => core.geom_idx,
        "vacua_estimate" => core.vac_est,
        "vacua_locations" => core.vac_id,
        "search" => core.search,
        "potential" => core.pot_data,
        "timings" => core.timings
    )
end

"""
    compute_axion_data(h11, np, cy, data_dir; kwargs...)

Compute the existing spectrum and vacua outputs for one geometry. The
vacuum-only path is used internally so the expensive stages are independently
timed; the returned spectrum and geometry fields remain backward compatible.
"""
function compute_axion_data(h11::Int, np::Int, cy::Int, data_dir::AbstractString;
        threshold::Float64=0.5, save::Bool=true, force::Bool=false,
        starts::Int=10_000, residual_tolerance::Float64=1e-10,
        merge_tolerance::Float64=1e-7, max_iterations::Int=200,
        method::Symbol=:legacy, max_branches::Int=1_000_000)
    geom_idx = CYAxiverse.structs.GeometryIndex(h11, np, cy)
    core = _vacua_core(geom_idx, data_dir; threshold, starts, residual_tolerance,
        merge_tolerance, max_iterations, method, max_branches)
    geometry_started_ns = time_ns()
    geom_data = CYAxiverse.read.geometry(geom_idx)
    _record_stage!(core.timings, "geometry_load_seconds", geometry_started_ns)
    spectrum_started_ns = time_ns()
    spectrum = CYAxiverse.generate.pq_spectrum(geom_idx)
    _record_stage!(core.timings, "spectrum_seconds", spectrum_started_ns)
    core.timings["total_seconds"] = (time_ns() - core.started_ns) / 1e9
    if save
        save_axion_data(geom_idx, spectrum, core.vac_est, core.vac_id;
            threshold, starts, residual_tolerance, merge_tolerance, max_iterations,
            method, max_branches, search_metadata=core.search,
            runtime_seconds=core.timings["total_seconds"], timings=core.timings, force)
    end
    Dict(
        "geom_idx" => geom_idx,
        "spectrum" => spectrum,
        "vacua_estimate" => core.vac_est,
        "vacua_locations" => core.vac_id,
        "search" => core.search,
        "potential" => core.pot_data,
        "geometry" => geom_data,
        "timings" => core.timings
    )
end

end # !isdefined(@__MODULE__, :VACUA_PIPELINE_VERSION)

# Command-Line Interface Execution
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 4
        println("Usage: julia run_axion_analysis.jl <h11> <np> <cy> <data_dir>")
        exit(1)
    end

    h11_in   = parse(Int, ARGS[1])
    np_in    = parse(Int, ARGS[2])
    cy_in    = parse(Int, ARGS[3])
    dir_in   = ARGS[4]

    println("==================================================")
    @printf("Processing Geometry: h11 = %d, np = %d, cy = %d\n", h11_in, np_in, cy_in)
    @printf("Data Directory: %s\n", dir_in)
    println("==================================================")

    results = compute_axion_data(h11_in, np_in, cy_in, dir_in)

    # --- Print Spectra Results ---
    println("\n[+] Axion Spectrum Summary:")
    println("  - Mass Eigenvalues (log10 eV):")
    println("    ", results["spectrum"].m)
    println("  - Decay Constants f_K (log10 M_Planck):")
    println("    ", results["spectrum"].fK)
    println("  - Sequential PQ Decay Quantities f (stored as fpert):")
    println("    ", results["spectrum"].f)
    
    # --- Print Vacua Results ---
    println("\n[+] Vacua Statistics & Locations:")
    println("  - Estimated Total Vacua Count: ", results["vacua_estimate"].vac)
    println("  - Is Qhat Square Matrix: ", results["vacua_estimate"].issquare == 1 ? "Yes" : "No")
    
    if haskey(results["vacua_locations"], "θ̃∥")
        println("  - Minima Coordinates Matrix (θ̃∥):")
        display(results["vacua_locations"]["θ̃∥"])
    elseif haskey(results["vacua_locations"], "vac")
        println("  - Verified Vacua Count: ", results["vacua_locations"]["vac"])
    end
    println("==================================================")
end
# julia run_axion_analysis.jl 10 20 1 "./my_data_dir"
