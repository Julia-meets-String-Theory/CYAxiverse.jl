#!/usr/bin/env julia

"""Exercise the script-level inflation screening call contract.

This driver intentionally contains the generic orchestration. It calls the
existing package APIs but does not add a generic scan function to the package.
The current contract is:

1. `read.potential(GeometryIndex)` loads `L`, `Q`, and `K`.
2. `generate.LQtilde(Q, L)` selects the leading independent charges.
3. `generate.instanton_hierarchy_diagnostics(L)` supplies a cheap hierarchy
   diagnostic.
4. `generate.leading_hessian_mass_basis_float64(K, Ltilde, Qtilde)` supplies a
   cheap Float64 mass-basis diagnostic.
5. `generate.foreach_leading_critical_branch(selected; max_branches=...)`
   streams bounded leading branches for full-potential screening.  An
   optional leading-index range enables deterministic low-index discovery.

The shared script helper uses log-shifted amplitudes so that screening ratios
remain finite for hierarchically suppressed instantons. It is deliberately
separate from the package's physical trajectory solver; the generic
trajectory/refinement call is not locked down yet.

Usage:

```text
julia --project=. scripts/inflation_scan_contract.jl \
    --data-dir DATA_ROOT --geometry H,P,F [--geometry H,P,F ...]
```
"""

include(joinpath(@__DIR__, "inflation_scan_common.jl"))

function _usage()
    println("Usage: julia --project=. scripts/inflation_scan_contract.jl " *
        "--data-dir DATA_ROOT --geometry H,P,F [--geometry H,P,F ...] " *
        "[--max-branches N] [--negative-mode-range K[:K]] " *
        "[--max-negative-modes K]")
end

function _parse_args(args)
    data_dir = get(ENV, "CYAXIVERSE_DATA_DIR", "")
    geometries = GeometryIndex[]
    max_branches = 1_000_000
    negative_mode_range = nothing
    max_negative_modes = nothing
    index = 1
    while index <= length(args)
        arg = args[index]
        if arg in ("--help", "-h")
            _usage()
            exit(0)
        elseif arg in ("--data-dir", "--geometry", "--max-branches",
                "--negative-mode-range", "--max-negative-modes")
            index == length(args) && error("missing value for $arg")
            value = args[index + 1]
            if arg == "--data-dir"
                data_dir = value
            elseif arg == "--geometry"
                parts = parse.(Int, split(value, ','))
                length(parts) == 3 || error("--geometry must be H,P,F")
                push!(geometries, GeometryIndex(parts...))
            else
                if arg == "--max-branches"
                    max_branches = parse(Int, value)
                elseif arg == "--negative-mode-range"
                    negative_mode_range = inflation_parse_negative_mode_range(value)
                else
                    max_negative_modes = parse(Int, value)
                end
            end
            index += 2
        else
            error("unknown argument $arg")
        end
    end
    isempty(geometries) && error("at least one --geometry H,P,F is required")
    max_branches > 0 || error("--max-branches must be positive")
    negative_mode_range === nothing || max_negative_modes === nothing ||
        error("use only one of --negative-mode-range and --max-negative-modes")
    (; data_dir=CYAxiverse.filestructure.resolve_data_dir(data_dir), geometries, max_branches,
       negative_mode_range, max_negative_modes)
end

function main(args)
    options = _parse_args(args)
    ENV["CYAXIVERSE_DATA_DIR"] = options.data_dir
    for geom_idx in options.geometries
        summary = try
            run_geometry(geom_idx; max_branches=options.max_branches,
                negative_mode_range=options.negative_mode_range,
                max_negative_modes=options.max_negative_modes)
        catch error
            failure = _scan_prep_error_status(error)
            (; contract_version=INFLATION_SCAN_CONTRACT_VERSION,
               h11=geom_idx.h11, polytope=geom_idx.polytope, frst=geom_idx.frst,
               status=failure.status, error=failure.message)
        end
        println(summary)
        flush(stdout)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
