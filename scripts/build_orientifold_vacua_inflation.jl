"""Phase 3 (vacua + inflation) driver for the h11=2 orientifold axiverse database.

For every `cyax.h5` under a given `h11_002` root, this script persists:

- `vacua_pipeline`: the existing reduced-JLM vacua count, computed via the
  package's own `scripts/vacua_pipeline.jl` (`compute_vacua_data`) with
  `method=:auto` passed explicitly. The default method on that engine is
  `:legacy`, which throws a `BoundsError` on these deeply-suppressed-instanton
  geometries (instanton log10 scales spanning roughly -111 to -654); `:auto`
  (exact determinant -> selected branch set -> reduced JLM finite search)
  completes cleanly. This script does not change that engine's default --
  it only ever calls it with `method=:auto`.
- `inflation/catastrophes`: the Hessian negative-mode classification of the
  leading critical branches, via `scripts/inflation_scan_common.jl`'s
  `run_geometry`. `leading_minima_count` (negative_modes==0) is the same
  quantity `vacua_pipeline`'s `:auto` path independently arrives at via a
  different algorithm (recorded as a cross-check, not the persisted vacua
  count); `saddle_count` (negative_modes>0 among the classified branches) is
  the catastrophe/tachyonic-branch count.
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

include(joinpath(@__DIR__, "vacua_pipeline.jl"))
include(joinpath(@__DIR__, "inflation_scan_common.jl"))
# `inflation_candidate_refinement.jl` has no self-guard against repeated
# `include` (unlike the two files above); guard it here so this script stays
# safe to `include` from a session (e.g. the test suite) that already loaded
# it, per the package's documented idempotent-include convention.
isdefined(@__MODULE__, :scan_geometry_for_inflation) ||
    include(joinpath(@__DIR__, "inflation_candidate_refinement.jl"))

const INFLATION_GROUP_SCHEMA_VERSION = "cyaxiverse-phase3-orientifold-inflation-1.0"

function _phase3_git_commit()
    try
        readchomp(`git -C $(dirname(@__DIR__)) rev-parse HEAD`)
    catch
        "unknown"
    end
end

"""Return whether a geometry file already carries an `inflation` group."""
function _has_inflation_group(path)
    isfile(path) || return false
    h5open(path, "r") do file
        haskey(file, "inflation")
    end
end

"""Write the catastrophe + efold results into a new `inflation` group.

Mirrors `save_axion_data`'s atomic temp-copy-then-rename pattern: never
mutates the target file in place, and never overwrites an existing
`inflation` group unless `force=true`.
"""
function write_inflation_group!(geom_idx, catastrophe_outcome, efold_outcome;
        catastrophe_settings, efold_settings, force::Bool=false)
    target = CYAxiverse.filestructure.cyax_file(geom_idx)
    isfile(target) || throw(ArgumentError("geometry file does not exist: $target"))
    !_has_inflation_group(target) || force ||
        throw(ArgumentError("inflation group already exists; pass force=true to replace $target"))

    temporary = string(target, ".inflation.tmp-", getpid(), "-", time_ns())
    cp(target, temporary; force=true)
    try
        h5open(temporary, "r+") do file
            haskey(file, "inflation") && HDF5.delete_object(file, "inflation")
            group = create_group(file, "inflation")
            group["schema_version"] = INFLATION_GROUP_SCHEMA_VERSION
            group["julia_version"] = string(VERSION)
            group["git_revision"] = _phase3_git_commit()
            group["completed_at"] = string(Dates.now())

            catastrophes = create_group(group, "catastrophes")
            catastrophes["method"] = "leading_critical_branch_hessian_classification"
            catastrophes["source"] = "scripts/inflation_scan_common.jl:run_geometry"
            if catastrophe_outcome.status == :success
                r = catastrophe_outcome.result
                catastrophes["status"] = "completed"
                catastrophes["leading_minima_count", deflate=9] = r.leading_minima_count
                catastrophes["saddle_count", deflate=9] = r.saddle_count
                catastrophes["catastrophes_present", deflate=9] = Int(r.saddle_count > 0)
                catastrophes["branch_count", deflate=9] = r.branch_count
                catastrophes["qtilde_det", deflate=9] = Int(r.qtilde_det)
                catastrophes["mass_min", deflate=9] = r.mass_min
                catastrophes["mass_max", deflate=9] = r.mass_max
                catastrophes["negative_mass_count", deflate=9] = r.negative_mass_count
                catastrophes["candidate_slowroll_saddles", deflate=9] = r.candidate_slowroll_saddles
                catastrophes["error"] = ""
            else
                catastrophes["status"] = string(catastrophe_outcome.status)
                catastrophes["leading_minima_count", deflate=9] = -1
                catastrophes["saddle_count", deflate=9] = -1
                catastrophes["catastrophes_present", deflate=9] = -1
                catastrophes["branch_count", deflate=9] = -1
                catastrophes["qtilde_det", deflate=9] = -1
                catastrophes["mass_min", deflate=9] = NaN
                catastrophes["mass_max", deflate=9] = NaN
                catastrophes["negative_mass_count", deflate=9] = -1
                catastrophes["candidate_slowroll_saddles", deflate=9] = -1
                catastrophes["error"] = catastrophe_outcome.message
            end
            catastrophe_meta = create_group(catastrophes, "metadata")
            for (name, value) in pairs(catastrophe_settings)
                catastrophe_meta[string(name)] = value isa Symbol ? string(value) :
                    value === nothing ? "none" : value
            end

            efolds = create_group(group, "efolds")
            efolds["method"] = "bounded_stationary_point_refinement_then_gradient_flow"
            efolds["source"] =
                "scripts/inflation_candidate_refinement.jl:scan_geometry_for_inflation"
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
                    trajectories["flow_status", deflate=9] =
                        [string(row.flow_status) for row in rows]
                    trajectories["flow_end_event", deflate=9] =
                        [string(row.flow_end_event) for row in rows]
                    trajectories["flow_efolds", deflate=9] =
                        Float64[Float64(row.flow_efolds) for row in rows]
                    trajectories["flow_slow_roll_efolds", deflate=9] =
                        Float64[Float64(row.flow_slow_roll_efolds) for row in rows]
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
function _pipeline2_complete(path::AbstractString)
    isfile(path) || return false
    h5open(path, "r") do file
        haskey(file, "vacua_pipeline") && haskey(file, "inflation")
    end
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

    results = NamedTuple[]
    skipped_count = 0
    for (index, (np, cy)) in enumerate(indices)
        geom_idx = CYAxiverse.structs.GeometryIndex(h11, np, cy)
        if skip_existing && !force &&
                _pipeline2_complete(CYAxiverse.filestructure.cyax_file(geom_idx))
            skipped_count += 1
            continue
        end
        print("[$index/$(length(indices))] h11=$h11 np=$np cy=$cy: ")

        vacua_status = :skipped
        vacua_vac = -1
        vacua_error = ""
        try
            vacua_result = compute_vacua_data(geom_idx, data_dir; method=:auto,
                save=true, force=force, starts=10_000)
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
        catastrophe_label = catastrophe_outcome.status == :success ?
            "$(catastrophe_outcome.status)(saddles=$(catastrophe_outcome.result.saddle_count))" :
            string(catastrophe_outcome.status)
        print("catastrophes=$catastrophe_label ")

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
            catastrophe_settings, efold_settings, force)

        push!(results, (; h11, np, cy, vacua_status, vacua_vac, vacua_error,
            catastrophe_status=catastrophe_outcome.status,
            saddle_count=catastrophe_outcome.status == :success ?
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
    println("catastrophes present (saddle_count>0): ",
        count(r -> r.saddle_count > 0, results))
    println("efold search completed: ",
        count(r -> r.efold_search_status == :completed, results))
    println("geometries with >=1 qualified efold trajectory: ",
        count(r -> r.n_qualified_efolds > 0, results))
end
