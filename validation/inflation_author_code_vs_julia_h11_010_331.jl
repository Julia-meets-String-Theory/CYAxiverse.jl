#!/usr/bin/env julia

"""Run the archived author code against a second deterministic h11=10 case.

The Julia package writes the actual physical fixed/full outputs to a temporary
exchange directory. The companion Python bridge then imports and calls the
archived ``geometric_quantities`` and ``Camcode_full_2.dim_reductor`` routines.
"""

using DelimitedFiles

include(joinpath(@__DIR__, "..", "scripts", "inflation_scale_continuation.jl"))

const DATA_ROOT = get(ENV, "CYAXIVERSE_DATA_DIR",
    normpath(joinpath(@__DIR__, "..", "..", "data")))
const GEOMETRY = GeometryIndex(10, 331, 1)
const AUTHOR_SOURCE = "/Users/vmehta/Documents/CYAxiverse/cyaxiverse/" *
    "CN_Axiverse_code/ks_axiverse_python_collaborator/src"
const DEFAULT_OUTPUT = "/private/tmp/cyaxiverse-author-bridge-h11-010-331"
const AUTHOR_PYTHON = get(ENV, "CYAXIVERSE_AUTHOR_PYTHON",
    "/Users/vmehta/.julia/conda/3/x86_64/bin/python")

function export_package_inputs(output::AbstractString)
    mkpath(output)
    loaded = CYAxiverse.read.oriented_potential(GEOMETRY)
    geometric = CYAxiverse.read.geometry(GEOMETRY)
    leading_count = _pilot_author_term_count(size(loaded.Q, 2))
    writedlm(joinpath(output, "tau.txt"), geometric.τ_volumes)
    writedlm(joinpath(output, "kinv.txt"), geometric.kinv)
    writedlm(joinpath(output, "author_charges.txt"), geometric.glsm_charges)
    writedlm(joinpath(output, "qprime.txt"),
        permutedims(loaded.Q[:, 1:leading_count]))
    open(joinpath(output, "cy_volume.txt"), "w") do io
        write(io, string(geometric.cy_volume))
    end
    for (name, scale) in (("0p9", 0.9), ("1p0", 1.0), ("1p1", 1.1))
        for mode in (:fixed, :full)
            scaled = pilot_scaled_inputs(loaded.Q, loaded.L, loaded.K, scale;
                scale_status=:physical, geometry=geometric,
                volume_normalization=mode)
            writedlm(joinpath(output, string(mode, "_", name, "_Q.txt")),
                scaled.Q)
            writedlm(joinpath(output, string(mode, "_", name, "_L.txt")),
                scaled.L)
        end
    end
    (; leading_count, term_count=size(loaded.Q, 2), cy_volume=geometric.cy_volume)
end

function main()
    output = get(ENV, "CYAXIVERSE_AUTHOR_BRIDGE_OUTPUT", DEFAULT_OUTPUT)
    ENV["CYAXIVERSE_DATA_DIR"] = DATA_ROOT
    metadata = export_package_inputs(output)
    bridge = joinpath(@__DIR__, "run_author_code_coefficient_bridge.py")
    run(`$AUTHOR_PYTHON $bridge --input-dir $output --author-src $AUTHOR_SOURCE`)
    println("package geometry=$(GEOMETRY.h11),$(GEOMETRY.polytope),$(GEOMETRY.frst) " *
        "metadata=$(metadata) output=$output")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
