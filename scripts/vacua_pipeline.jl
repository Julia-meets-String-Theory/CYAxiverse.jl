using CYAxiverse
using HDF5
using LinearAlgebra
using Printf
using Dates

# This file is included by both the test harness and the batch runner.  Keep
# the helper definitions behind a module-local guard so a second include is a
# no-op while direct invocation still reaches the CLI below.
if !isdefined(@__MODULE__, :VACUA_PIPELINE_VERSION)

const VACUA_PIPELINE_VERSION = "1"

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

"""Return whether a completed result matches an optional saved configuration."""
function _has_pipeline_result(path; config=nothing)
    isfile(path) || return false
    h5open(path, "r") do file
        haskey(file, "vacua_pipeline/metadata/status") || return false
        read(file["vacua_pipeline/metadata/status"]) == "completed" || return false
        config === nothing && return true
        metadata = file["vacua_pipeline/metadata"]
        required = (("pipeline_version", config.pipeline_version),
            ("threshold", config.threshold), ("starts", config.starts),
            ("residual_tolerance", config.residual_tolerance),
            ("merge_tolerance", config.merge_tolerance),
            ("max_iterations", config.max_iterations), ("method", config.method),
            ("max_branches", config.max_branches))
        all(name -> haskey(metadata, name[1]) && read(metadata[name[1]]) == name[2], required)
    end
end

"""Return whether an HDF5 geometry file contains a vacua pipeline group."""
function _has_pipeline_group(path)
    isfile(path) || return false
    h5open(path, "r") do file
        haskey(file, "vacua_pipeline")
    end
end

"""Write solver, provenance, status, and search metadata to an HDF5 group."""
function _write_metadata!(group, config; status, estimate_status, verification_status,
    runtime_seconds, solver_status=status, search_metadata=nothing, error_message="")
    metadata = create_group(group, "metadata")
    for (name, value) in pairs(config)
        metadata[string(name)] = value
    end
    metadata["status"] = status
    metadata["solver_status"] = solver_status
    metadata["estimate_status"] = estimate_status
    metadata["verification_status"] = verification_status
    metadata["julia_version"] = string(VERSION)
    metadata["git_revision"] = _git_revision()
    metadata["runtime_seconds"] = runtime_seconds
    metadata["completed_at"] = string(Dates.now())
    metadata["error"] = error_message
    if search_metadata !== nothing
        for (name, value) in pairs(search_metadata)
            metadata[string(name)] = value
        end
    end
end

"""Dispatch one geometry to the selected vacua search method."""
function _search_vacua(geom_idx, pot_data; threshold, starts, residual_tolerance,
        merge_tolerance, max_iterations, method, max_branches)
    Q, L = pot_data.Q, pot_data.L
    if method == :legacy
        estimate = CYAxiverse.generate.vacua_estimate(
            geom_idx; threshold=threshold)
        locations = CYAxiverse.generate.vacua_id(L, Q;
            threshold=threshold, runs=starts)
        search = (; search_method="legacy", search_classification="finite_search_lower_bound",
            minimum_count=locations["vac"], multiplicity=1.0,
            critical_count=-1, branch_count=-1, det_Qtilde=-1,
            search_status="completed")
        return estimate, locations, search
    elseif method == :leading_branches
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
            search_classification="certified_selected_branch_set",
            minimum_count=branches.leading_minima_count, multiplicity=1.0,
            critical_count=branches.branch_count, branch_count=branches.branch_count,
            det_Qtilde=branches.det_Qtilde, search_status="completed")
        return estimate, locations, search
    elseif method == :reduced_jlm
        problem = CYAxiverse.jlm_reduced.prepare(Q, L; threshold=threshold)
        minimum = CYAxiverse.jlm_reduced.minimize(problem; starts=starts,
            residual_tolerance=residual_tolerance,
            merge_tolerance=merge_tolerance, max_iterations=max_iterations)
        square = minimum isa CYAxiverse.structs.Min_JLM_Square
        estimate = square ? (; vac=minimum.N_min, issquare=1) :
            (; vac=minimum.N_min, issquare=0, extrarows=minimum.extra_rows)
        locations = Dict{String, Int}("vac" => minimum.N_min)
        search = (; search_method="reduced_jlm",
            search_classification=square ? "exact_determinant_branch" :
                "finite_search_lower_bound",
            minimum_count=minimum.N_min, multiplicity=problem.multiplicity,
            critical_count=-1, branch_count=-1, det_Qtilde=minimum.det_QTilde,
            search_status="completed")
        return estimate, locations, search
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
    search_metadata=nothing, runtime_seconds::Float64=0.0, force::Bool=false)
    target = CYAxiverse.filestructure.cyax_file(geom_idx)
    isfile(target) || throw(ArgumentError("geometry file does not exist: $target"))
    config = _pipeline_config(; threshold, starts, residual_tolerance,
        merge_tolerance, max_iterations, method, max_branches)
    !_has_pipeline_group(target) || force ||
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
            elseif search_metadata.search_classification == "exact_determinant_branch"
                "verified"
            elseif search_metadata.search_classification == "certified_selected_branch_set"
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
                    search_metadata.search_status, search_metadata)
        end
        mv(temporary, target; force=true)
    catch
        isfile(temporary) && rm(temporary; force=true)
        rethrow()
    end
    nothing
end

"""
    compute_axion_data(h11, np, cy, data_dir; kwargs...)

Compute legacy spectrum and vacua outputs for one geometry. The numerical
search parameters are recorded in HDF5 metadata; saving is disabled with
`save=false` and existing pipeline output is protected unless `force=true`.
"""
function compute_axion_data(h11::Int, np::Int, cy::Int, data_dir::String;
        threshold::Float64=0.5, save::Bool=true, force::Bool=false,
    starts::Int=10_000, residual_tolerance::Float64=1e-10,
        merge_tolerance::Float64=1e-7, max_iterations::Int=200,
        method::Symbol=:legacy, max_branches::Int=1_000_000)
    started_at = time()
    ENV["CYAXIVERSE_DATA_DIR"] = _validate_data_dir(data_dir)
    threshold >= 0 || throw(ArgumentError("threshold must be nonnegative"))
    starts > 0 || throw(ArgumentError("starts must be positive"))
    residual_tolerance > 0 || throw(ArgumentError("residual_tolerance must be positive"))
    merge_tolerance > 0 || throw(ArgumentError("merge_tolerance must be positive"))
    max_iterations > 0 || throw(ArgumentError("max_iterations must be positive"))
    method in (:legacy, :leading_branches, :reduced_jlm) ||
        throw(ArgumentError("unsupported method: $method"))
    max_branches > 0 || throw(ArgumentError("max_branches must be positive"))

    # 1. Construct Geometry Index
    geom_idx = CYAxiverse.structs.GeometryIndex(h11, np, cy)
    
    pot_data = CYAxiverse.read.potential(geom_idx)
    _validate_potential(geom_idx, pot_data)
    geom_data = CYAxiverse.read.geometry(geom_idx)
    
    # 3. Calculate the PQ mass-basis spectrum and quartic couplings.
    spectrum = CYAxiverse.generate.pq_spectrum(geom_idx)
    
    # 4. Calculate Vacua Statistics & Locations
    vac_est, vac_id, search = _search_vacua(geom_idx, pot_data; threshold, starts,
        residual_tolerance, merge_tolerance, max_iterations, method, max_branches)
    if save
        save_axion_data(geom_idx, spectrum, vac_est, vac_id; threshold=threshold,
            starts, residual_tolerance, merge_tolerance, max_iterations, method,
            max_branches, search_metadata=search,
            runtime_seconds=time() - started_at, force)
    end
    
    return Dict(
        "geom_idx" => geom_idx,
        "spectrum" => spectrum,
        "vacua_estimate" => vac_est,
        "vacua_locations" => vac_id,
        "search" => search,
        "potential" => pot_data,
        "geometry" => geom_data
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
