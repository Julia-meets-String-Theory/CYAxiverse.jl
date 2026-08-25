"""Phase 3 (vacua + inflation) driver for the h11=2 orientifold axiverse database.

For every `cyax.h5` under a given `h11_002` root, this script persists:

- `vacua_pipeline`: the hierarchy-truncated vacua estimate, computed via the
  package's own `scripts/vacua_pipeline.jl` (`compute_vacua_data`) with
  `method=:auto` passed explicitly. The default method on that engine is
  `:legacy`, which throws a `BoundsError` on these deeply-suppressed-instanton
  geometries (instanton log10 scales spanning roughly -111 to -654); `:auto`
  (exact determinant -> selected branch set -> legacy finite-search fallback)
  completes cleanly. This script does not change that engine's default --
  it only ever calls it with `method=:auto`.
- `inflation/catastrophes` (legacy-compatible group name): the Hessian
  negative-mode classification of the
  leading critical branches, via `scripts/inflation_scan_common.jl`'s
  `run_geometry`. `leading_minima_count` (negative_modes==0) is the same
  quantity `vacua_pipeline`'s `:auto` path independently arrives at via a
  different algorithm (recorded as a cross-check, not the persisted vacua
  count); `leading_branch_saddle_count` (negative_modes>0 among the classified
  branches) is the leading-branch saddle count.
- `inflation/efolds`: candidate inflationary trajectories from
  `scripts/inflation_candidate_refinement.jl`'s `scan_geometry_for_inflation`
  (bounded stationary-point refinement at Float64 and arbitrary precision,
  then gradient flow from each accepted candidate's negative physical mass
  mode in both displacement directions).

Every write is append-only: an existing `vacua_pipeline` or `inflation` group
blocks the corresponding write unless `--force` is passed, and both writers
follow the same temp-copy-then-atomic-rename pattern `save_axion_data` uses,
so a geometry's file is never left partially written.
"""

using CYAxiverse
using LinearAlgebra
using Statistics
using HDF5
using Printf
using Dates
using Random
using SHA

include(joinpath(@__DIR__, "vacua_pipeline.jl"))
include(joinpath(@__DIR__, "inflation_scan_common.jl"))
# `inflation_candidate_refinement.jl` has no self-guard against repeated
# `include` (unlike the two files above); guard it here so this script stays
# safe to `include` from a session (e.g. the test suite) that already loaded
# it, per the package's documented idempotent-include convention.
isdefined(@__MODULE__, :scan_geometry_for_inflation) ||
    include(joinpath(@__DIR__, "inflation_candidate_refinement.jl"))

const INFLATION_GROUP_SCHEMA_VERSION = "cyaxiverse-phase3-orientifold-inflation-2.2"
const INFLATION_LEGACY_SCHEMA_VERSION = "cyaxiverse-phase3-orientifold-inflation-1.0"
const INFLATION_FLOW_ENCODING_SCHEMA_VERSION = "cyaxiverse-inflation-flow-decimal-string-1.0"
const INFLATION_DOMAIN_CERTIFICATE_VERSION = "physical-domain-certificate-1"
const INFLATION_PHYSICAL_UNITS_CONTRACT = "M_s=M_Pl;k=dimensionless"

"""Hash the complete Pipeline 2 configuration contract."""
_phase3_config_digest(value) = bytes2hex(sha256(repr(value)))

"""Return a fixed-width UTF-8 byte matrix and lengths for strings.

HDF5 cannot combine variable-length Julia `String` vectors with compression.
Store the bytes explicitly instead: each column is one UTF-8 string, padded
with zero bytes, and the companion lengths dataset makes the representation
lossless and replayable without a numeric narrowing conversion.
"""
function _string_bytes(values)
    encoded = [Vector{UInt8}(codeunits(string(value))) for value in values]
    lengths = Int[length(bytes) for bytes in encoded]
    width = isempty(lengths) ? 0 : maximum(lengths)
    bytes = zeros(UInt8, width, length(encoded))
    for (column, value) in enumerate(encoded)
        isempty(value) || (bytes[1:length(value), column] = value)
    end
    bytes, lengths
end

function _write_string_bytes!(group, name, values)
    bytes, lengths = _string_bytes(values)
    group[name, deflate=9] = bytes
    group[string(name, "_lengths"), deflate=9] = lengths
end

function _phase3_git_commit()
    try
        readchomp(`git -C $(dirname(@__DIR__)) rev-parse HEAD`)
    catch
        "unknown"
    end
end

"""Return whether a geometry file already carries an `inflation` group."""
function _has_inflation_group(path; config_digest=nothing)
    isfile(path) || return false
    h5open(path, "r") do file
        required = ("schema_version", "status", "terminal_status",
            "configuration_digest", "git_revision", "domain_certificate_version",
            "scale_status", "domain_status", "fixed_point_status",
            "trajectory_status", "coverage_status", "moduli_status",
            "phase_convention", "units", "physical_units_contract",
            "normalization", "source_identity",
            "precision_bits")
        all(name -> haskey(file, "inflation/$name"), required) || return false
        read(file["inflation/status"]) == "completed" || return false
        read(file["inflation/physical_units_contract"]) ==
            INFLATION_PHYSICAL_UNITS_CONTRACT || return false
        config_digest === nothing && return true
        haskey(file, "inflation/configuration_digest") || return false
        read(file["inflation/configuration_digest"]) == config_digest
    end
end

"""Return whether an existing inflation group must not be replaced implicitly."""
function _inflation_write_blocked(path)
    isfile(path) || return false
    h5open(path, "r") do file
        haskey(file, "inflation") || return false
        # Preserve every existing group, including failed and legacy groups.
        # Replacing one requires the caller to state force=true explicitly.
        true
    end
end

"""Write the catastrophe + efold results into a new `inflation` group.

Mirrors `save_axion_data`'s atomic temp-copy-then-rename pattern: never
mutates the target file in place, and never overwrites an existing
`inflation` group unless `force=true`.
"""
function write_inflation_group!(geom_idx, catastrophe_outcome, efold_outcome;
        catastrophe_settings, efold_settings, force::Bool=false,
        pipeline_config_digest=nothing)
    target = CYAxiverse.filestructure.cyax_file(geom_idx)
    isfile(target) || throw(ArgumentError("geometry file does not exist: $target"))
    config_digest = something(pipeline_config_digest,
        _phase3_config_digest((; catastrophe_settings, efold_settings)))
    !_inflation_write_blocked(target) || force ||
        throw(ArgumentError("inflation group already exists; pass force=true to replace $target"))

    temporary = string(target, ".inflation.tmp-", getpid(), "-", time_ns())
    phase3_status = catastrophe_outcome.status == :success &&
        efold_outcome.status == :success ? "completed" : "failed"
    phase3_coverage_status = efold_outcome.status == :success ?
        string(efold_outcome.result.refinement.search.coverage_status) :
        string(efold_outcome.status)
    phase3_source_identity = string("scripts/build_orientifold_vacua_inflation.jl@",
        _phase3_git_commit())
    cp(target, temporary; force=true)
    try
        h5open(temporary, "r+") do file
            haskey(file, "inflation") && HDF5.delete_object(file, "inflation")
            group = create_group(file, "inflation")
            group["schema_version"] = INFLATION_GROUP_SCHEMA_VERSION
            group["configuration_digest"] = config_digest
            group["status"] = phase3_status
            group["terminal_status"] = phase3_status
            group["julia_version"] = string(VERSION)
            group["git_revision"] = _phase3_git_commit()
            group["completed_at"] = string(Dates.now())
            group["scale_status"] = "homotopy_only"
            group["domain_certificate_version"] = INFLATION_DOMAIN_CERTIFICATE_VERSION
            group["domain_status"] = "out_of_model"
            group["fixed_point_status"] = catastrophe_outcome.status == :success ?
                "homotopy_only" : "not_run"
            group["trajectory_status"] = efold_outcome.status == :success ?
                "homotopy_only" : "not_run"
            group["coverage_status"] = phase3_coverage_status
            group["moduli_status"] = "not_established"
            group["phase_convention"] = "not_persisted"
            group["units"] = "not_persisted"
            group["physical_units_contract"] = INFLATION_PHYSICAL_UNITS_CONTRACT
            group["normalization"] = "homotopy_only"
            group["source_identity"] = phase3_source_identity
            group["precision_bits"] = Int(efold_settings.precision_bits)
            group["claim_boundary"] =
                "homotopy-only diagnostics; physical domain certificate not passed"

            catastrophes = create_group(group, "catastrophes")
            catastrophes["method"] = "leading_critical_branch_hessian_classification"
            catastrophes["source"] = "scripts/inflation_scan_common.jl:run_geometry"
            catastrophes["domain_certificate_version"] = INFLATION_DOMAIN_CERTIFICATE_VERSION
            catastrophes["scale_status"] = "homotopy_only"
            catastrophes["domain_status"] = "out_of_model"
            catastrophes["fixed_point_status"] = "homotopy_only"
            catastrophes["trajectory_status"] = "not_run"
            catastrophes["coverage_status"] = "not_applicable"
            catastrophes["moduli_status"] = "not_established"
            catastrophes["phase_convention"] = "not_persisted"
            catastrophes["units"] = "not_persisted"
            catastrophes["physical_units_contract"] = INFLATION_PHYSICAL_UNITS_CONTRACT
            catastrophes["normalization"] = "homotopy_only"
            catastrophes["source_identity"] = phase3_source_identity
            catastrophes["precision_bits"] = Int(efold_settings.precision_bits)
            if catastrophe_outcome.status == :success
                r = catastrophe_outcome.result
                catastrophes["status"] = "completed"
                catastrophes["leading_minima_count", deflate=9] = r.leading_minima_count
                catastrophes["leading_branch_saddle_count", deflate=9] = r.saddle_count
                catastrophes["leading_branch_saddles_present", deflate=9] =
                    Int(r.saddle_count > 0)
                catastrophes["branch_count", deflate=9] = r.branch_count
                catastrophes["qtilde_det", deflate=9] = Int(r.qtilde_det)
                catastrophes["mass_min", deflate=9] = r.mass_min
                catastrophes["mass_max", deflate=9] = r.mass_max
                catastrophes["negative_mass_count", deflate=9] = r.negative_mass_count
                catastrophes["candidate_slowroll_saddles", deflate=9] = r.candidate_slowroll_saddles
                catastrophes["error"] = ""
                legacy_saddle_count = r.saddle_count
                legacy_saddles_present = Int(r.saddle_count > 0)
            else
                catastrophes["status"] = string(catastrophe_outcome.status)
                catastrophes["leading_minima_count", deflate=9] = -1
                catastrophes["leading_branch_saddle_count", deflate=9] = -1
                catastrophes["leading_branch_saddles_present", deflate=9] = -1
                catastrophes["branch_count", deflate=9] = -1
                catastrophes["qtilde_det", deflate=9] = -1
                catastrophes["mass_min", deflate=9] = NaN
                catastrophes["mass_max", deflate=9] = NaN
                catastrophes["negative_mass_count", deflate=9] = -1
                catastrophes["candidate_slowroll_saddles", deflate=9] = -1
                catastrophes["error"] = catastrophe_outcome.message
                legacy_saddle_count = -1
                legacy_saddles_present = -1
            end
            legacy = create_group(catastrophes, "legacy_v1")
            legacy["schema_version"] = INFLATION_LEGACY_SCHEMA_VERSION
            legacy["migration"] =
                "saddle_count -> leading_branch_saddle_count; catastrophes_present -> leading_branch_saddles_present"
            legacy["saddle_count", deflate=9] = legacy_saddle_count
            legacy["catastrophes_present", deflate=9] = legacy_saddles_present
            catastrophe_meta = create_group(catastrophes, "metadata")
            for (name, value) in pairs(catastrophe_settings)
                catastrophe_meta[string(name)] = value isa Symbol ? string(value) :
                    value === nothing ? "none" : value
            end

            efolds = create_group(group, "efolds")
            efolds["method"] = "bounded_stationary_point_refinement_then_gradient_flow"
            efolds["source"] =
                "scripts/inflation_candidate_refinement.jl:scan_geometry_for_inflation"
            efolds["domain_certificate_version"] = INFLATION_DOMAIN_CERTIFICATE_VERSION
            efolds["scale_status"] = "homotopy_only"
            efolds["domain_status"] = "out_of_model"
            efolds["fixed_point_status"] = "homotopy_only"
            efolds["trajectory_status"] = efold_outcome.status == :success ?
                "homotopy_only" : "not_run"
            efolds["moduli_status"] = "not_established"
            efolds["phase_convention"] = "not_persisted"
            efolds["units"] = "not_persisted"
            efolds["physical_units_contract"] = INFLATION_PHYSICAL_UNITS_CONTRACT
            efolds["normalization"] = "homotopy_only"
            efolds["source_identity"] = phase3_source_identity
            efolds["precision_bits"] = Int(efold_settings.precision_bits)
            if efold_outcome.status == :success
                scan = efold_outcome.result
                efolds["search_status"] = string(scan.refinement.search.search_status)
                efolds["coverage_status"] = string(scan.refinement.search.coverage_status)
                efolds["refinement_status"] = string(scan.refinement.search.refinement_status)
                efolds["n_candidates", deflate=9] = length(scan.refinement.candidates)
                efolds["n_flow_rows", deflate=9] = length(scan.flow.rows)
                efolds["n_qualified", deflate=9] = scan.flow.qualified
                efolds["error"] = ""
                rows = scan.flow.rows
                if !isempty(rows)
                    trajectories = create_group(efolds, "trajectories")
                    trajectories["flow_precision_bits", deflate=9] =
                        maximum(Int[row.precision_bits for row in rows])
                    trajectories["flow_precision_bits_by_row", deflate=9] =
                        Int[row.precision_bits for row in rows]
                    trajectories["flow_numeric_encoding"] = "decimal_string"
                    trajectories["flow_numeric_encoding_schema"] =
                        INFLATION_FLOW_ENCODING_SCHEMA_VERSION
                    _write_string_bytes!(trajectories, "flow_status",
                        [row.flow_status for row in rows])
                    _write_string_bytes!(trajectories, "flow_end_event",
                        [row.flow_end_event for row in rows])
                    # BigFloat values are persisted as decimal strings.  This
                    # is the smallest HDF5 representation that preserves the
                    # arbitrary-precision value without an implicit Float64
                    # conversion; precision_bits and the encoding schema make
                    # the representation replayable.
                    _write_string_bytes!(trajectories, "flow_efolds",
                        [row.flow_efolds for row in rows])
                    _write_string_bytes!(trajectories, "flow_slow_roll_efolds",
                        [row.flow_slow_roll_efolds for row in rows])
                    trajectories["flow_accepted", deflate=9] =
                        Int[Int(row.flow_accepted) for row in rows]
                    trajectories["candidate_index", deflate=9] =
                        Int[row.candidate_index for row in rows]
                    trajectories["displacement_sign", deflate=9] =
                        Int[row.displacement_sign === nothing ? 0 : row.displacement_sign
                            for row in rows]
                    trajectories["leading_negative_modes", deflate=9] =
                        Int[row.leading_negative_modes for row in rows]
                end
            else
                efolds["search_status"] = string(efold_outcome.status)
                efolds["coverage_status"] = "not_applicable"
                efolds["refinement_status"] = "not_started"
                efolds["n_candidates", deflate=9] = -1
                efolds["n_flow_rows", deflate=9] = -1
                efolds["n_qualified", deflate=9] = -1
                efolds["error"] = efold_outcome.message
            end
            efold_meta = create_group(efolds, "metadata")
            for (name, value) in pairs(efold_settings)
                efold_meta[string(name)] = value isa Symbol ? string(value) :
                    value === nothing ? "none" : value
            end
        end
        mv(temporary, target; force=true)
    catch
        isfile(temporary) && rm(temporary; force=true)
        rethrow()
    end
    nothing
end

"""Discover `(np, cy)` indices for every `cyax.h5` under one `h11_XXX` root."""
function discover_geometries(h11_dir::AbstractString)
    indices = Tuple{Int, Int}[]
    for np_name in sort(readdir(h11_dir))
        startswith(np_name, "np_") || continue
        np = parse(Int, np_name[4:end])
        np_dir = joinpath(h11_dir, np_name)
        for cy_name in sort(readdir(np_dir))
            startswith(cy_name, "cy_") || continue
            cy = parse(Int, cy_name[4:end])
            isfile(joinpath(np_dir, cy_name, "cyax.h5")) || continue
            push!(indices, (np, cy))
        end
    end
    indices
end

"""Return whether a geometry already has both Pipeline 2 groups written.

Lets `run_pipeline2` be re-invoked across several bounded foreground calls
for a large h11 population (each call picks up exactly where the previous
one left off) without needing an explicit index range: an already-complete
geometry is skipped rather than re-attempted (which would otherwise hit the
`vacua_pipeline`/`inflation` no-overwrite guards and be misreported as a
failure).
"""
function _pipeline2_complete(path::AbstractString; vacua_config=nothing,
        config_digest=nothing)
    isfile(path) || return false
    _has_pipeline_result(path; config=vacua_config) || return false
    _has_inflation_group(path; config_digest) || return false
    true
end

function run_pipeline2(data_dir::AbstractString; h11::Int=2, force::Bool=false,
        max_branches_catastrophe::Int=1_000_000, max_branches_efold::Int=100_000,
        precision_bits::Int=256, float_tolerance::Float64=1e-10,
        high_tolerance::Float64=1e-40, max_points::Int=1000,
        min_efolds::Real=50, max_efolds::Real=60,
        flow_step::Real=1e-3, flow_displacement::Real=1e-8,
        skip_existing::Bool=true)
    h11_dir = joinpath(data_dir, @sprintf("h11_%03d", h11))
    isdir(h11_dir) || throw(ArgumentError("h11 directory does not exist: $h11_dir"))
    ENV["CYAXIVERSE_DATA_DIR"] = data_dir

    indices = discover_geometries(h11_dir)
    println("discovered $(length(indices)) geometries under $h11_dir")

    catastrophe_settings = (; max_branches=max_branches_catastrophe)
    efold_settings = (; max_branches=max_branches_efold, precision_bits,
        float_tolerance, high_tolerance, max_points, min_efolds, max_efolds,
        flow_step, flow_displacement)
    vacua_config = _pipeline_config(; threshold=0.5, starts=10_000,
        residual_tolerance=1e-10, merge_tolerance=1e-7, max_iterations=200,
        method=:auto, max_branches=1_000_000)
    source_identity = _phase3_git_commit()
    pipeline_config_digest = _phase3_config_digest((; h11, vacua_config,
        catastrophe_settings, efold_settings, source_identity,
        inflation_schema=INFLATION_GROUP_SCHEMA_VERSION,
        domain_certificate_version=INFLATION_DOMAIN_CERTIFICATE_VERSION))

    results = NamedTuple[]
    skipped_count = 0
    for (index, (np, cy)) in enumerate(indices)
        geom_idx = CYAxiverse.structs.GeometryIndex(h11, np, cy)
        target_path = CYAxiverse.filestructure.cyax_file(geom_idx)
        if skip_existing && !force &&
                _pipeline2_complete(target_path;
                    vacua_config, config_digest=pipeline_config_digest)
            skipped_count += 1
            continue
        end
        if !force && (_pipeline_group_exists(target_path) ||
                _inflation_write_blocked(target_path))
            println("blocked: persisted vacua/inflation data exists; pass force=true")
            push!(results, (; h11, np, cy, vacua_status=:blocked,
                vacua_vac=-1,
                vacua_error="persisted pipeline data exists; explicit force=true required",
                leading_branch_status=:blocked,
                leading_branch_saddle_count=-1,
                leading_minima_count=-1,
                efold_search_status=:blocked,
                n_efold_candidates=-1,
                n_qualified_efolds=-1))
            continue
        end
        print("[$index/$(length(indices))] h11=$h11 np=$np cy=$cy: ")

        vacua_status = :skipped
        vacua_vac = -1
        vacua_error = ""
        try
            vacua_force = force
            vacua_result = compute_vacua_data(geom_idx, data_dir; method=:auto,
                save=true, force=vacua_force, starts=10_000)
            vacua_status = Symbol(vacua_result["search"].search_status)
            vacua_vac = vacua_result["vacua_estimate"].vac
        catch error
            vacua_status = :failed
            vacua_error = sprint(showerror, error)
        end
        print("vacua=$vacua_status($vacua_vac) ")

        catastrophe_outcome = try
            (; status=:success, result=run_geometry(geom_idx;
                max_branches=max_branches_catastrophe))
        catch error
            (; status=:failed, message=sprint(showerror, error))
        end
        leading_branch_label = catastrophe_outcome.status == :success ?
            "$(catastrophe_outcome.status)(leading_branch_saddles=$(catastrophe_outcome.result.saddle_count))" :
            string(catastrophe_outcome.status)
        print("leading_branch_saddles=$leading_branch_label ")

        efold_outcome = try
            (; status=:success, result=scan_geometry_for_inflation(geom_idx;
                max_branches=max_branches_efold, precision_bits, float_tolerance,
                high_tolerance, max_points, min_efolds, max_efolds,
                flow_step, flow_displacement))
        catch error
            (; status=:failed, message=sprint(showerror, error))
        end
        efold_label = efold_outcome.status == :success ?
            "$(efold_outcome.status)(qualified=$(efold_outcome.result.flow.qualified))" :
            string(efold_outcome.status)
        println("efolds=$efold_label")

        write_inflation_group!(geom_idx, catastrophe_outcome, efold_outcome;
            catastrophe_settings, efold_settings,
            pipeline_config_digest=pipeline_config_digest,
            force=force)

        push!(results, (; h11, np, cy, vacua_status, vacua_vac, vacua_error,
            leading_branch_status=catastrophe_outcome.status,
            leading_branch_saddle_count=catastrophe_outcome.status == :success ?
                catastrophe_outcome.result.saddle_count : -1,
            leading_minima_count=catastrophe_outcome.status == :success ?
                catastrophe_outcome.result.leading_minima_count : -1,
            efold_search_status=efold_outcome.status == :success ?
                efold_outcome.result.refinement.search.search_status : efold_outcome.status,
            n_efold_candidates=efold_outcome.status == :success ?
                length(efold_outcome.result.refinement.candidates) : -1,
            n_qualified_efolds=efold_outcome.status == :success ?
                efold_outcome.result.flow.qualified : -1))
    end
    skipped_count > 0 && println(
        "skipped $skipped_count/$(length(indices)) geometries already complete " *
        "(vacua_pipeline and inflation groups both present)")
    results
end

if abspath(PROGRAM_FILE) == @__FILE__
    data_dir = length(ARGS) >= 1 ? ARGS[1] :
        "/Users/vmehta/Documents/CYAxiverse/cyaxiverse/data/orientifold_axiverse_database_20260821"
    results = run_pipeline2(data_dir; h11=2)
    println("\n=== summary ===")
    println("geometries: ", length(results))
    println("vacua completed: ", count(r -> r.vacua_status == :completed, results))
    println("leading-branch saddles present (leading_branch_saddle_count>0): ",
        count(r -> r.leading_branch_saddle_count > 0, results))
    println("efold search completed: ",
        count(r -> r.efold_search_status == :completed, results))
    println("geometries with >=1 qualified efold trajectory: ",
        count(r -> r.n_qualified_efolds > 0, results))
end
