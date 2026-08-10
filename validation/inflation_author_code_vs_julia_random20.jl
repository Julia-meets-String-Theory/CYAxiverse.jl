#!/usr/bin/env julia

"""Compare the package with the archived author code on 20 random geometries.

The h11 value is sampled uniformly from 4:50 and then a geometry is sampled
uniformly from that h11 group, without duplicate geometry identities. The
fixed/full package outputs are passed to the Python bridge, which calls the
archived ``geometric_quantities`` and ``Camcode_full_2.dim_reductor`` routines
unchanged.
"""

using DelimitedFiles
using Printf
using Random

include(joinpath(@__DIR__, "..", "scripts", "inflation_scale_continuation.jl"))

const DATA_ROOT = get(ENV, "CYAXIVERSE_DATA_DIR",
    normpath(joinpath(@__DIR__, "..", "..", "data")))
const AUTHOR_SOURCE = "/Users/vmehta/Documents/CYAxiverse/cyaxiverse/" *
    "CN_Axiverse_code/ks_axiverse_python_collaborator/src"
const AUTHOR_PYTHON = get(ENV, "CYAXIVERSE_AUTHOR_PYTHON",
    "/Users/vmehta/.julia/conda/3/x86_64/bin/python")
const DEFAULT_OUTPUT = "/private/tmp/cyaxiverse-author-bridge-random20-h11-004-050"
const SAMPLE_SEED = 20260820
const SAMPLE_COUNT = 20
const H11_MIN = 4
const H11_MAX = 50

function _indexed_geometries(data_root::AbstractString, h11::Int)
    h11_dir = joinpath(data_root, @sprintf("h11_%03d", h11))
    isdir(h11_dir) || throw(ArgumentError("missing h11 directory: $h11_dir"))
    geometries = GeometryIndex[]
    for np_name in sort(readdir(h11_dir))
        startswith(np_name, "np_") || continue
        np_number = parse(Int, np_name[4:end])
        np_dir = joinpath(h11_dir, np_name)
        isdir(np_dir) || continue
        for cy_name in sort(readdir(np_dir))
            startswith(cy_name, "cy_") || continue
            cy_number = parse(Int, cy_name[4:end])
            geometry_dir = joinpath(np_dir, cy_name)
            isfile(joinpath(geometry_dir, "cyax.h5")) || continue
            push!(geometries, GeometryIndex(h11, np_number, cy_number))
        end
    end
    isempty(geometries) && throw(ArgumentError("no geometries found in $h11_dir"))
    geometries
end

function select_geometries(data_root::AbstractString; count::Int=SAMPLE_COUNT,
        h11_min::Int=H11_MIN, h11_max::Int=H11_MAX, seed::Int=SAMPLE_SEED)
    by_h11 = Dict(h11 => _indexed_geometries(data_root, h11)
        for h11 in h11_min:h11_max)
    rng = MersenneTwister(seed)
    selected = GeometryIndex[]
    seen = Set{Tuple{Int, Int, Int}}()
    while length(selected) < count
        h11 = rand(rng, h11_min:h11_max)
        candidates = by_h11[h11]
        candidate = candidates[rand(rng, eachindex(candidates))]
        key = (candidate.h11, candidate.polytope, candidate.frst)
        key in seen && continue
        push!(seen, key)
        push!(selected, candidate)
    end
    selected
end

function _scale_name(scale::Float64)
    replace(string(scale), "." => "p")
end

function export_package_inputs(geometry::GeometryIndex, output::AbstractString)
    mkpath(output)
    loaded = CYAxiverse.read.oriented_potential(geometry)
    geometric = CYAxiverse.read.geometry(geometry)
    leading_count = _pilot_author_term_count(size(loaded.Q, 2))
    writedlm(joinpath(output, "tau.txt"), geometric.τ_volumes)
    writedlm(joinpath(output, "kinv.txt"), geometric.kinv)
    writedlm(joinpath(output, "author_charges.txt"), geometric.glsm_charges)
    writedlm(joinpath(output, "qprime.txt"),
        permutedims(loaded.Q[:, 1:leading_count]))
    open(joinpath(output, "cy_volume.txt"), "w") do io
        write(io, string(geometric.cy_volume))
    end
    open(joinpath(output, "geometry_index.txt"), "w") do io
        write(io, "$(geometry.h11),$(geometry.polytope),$(geometry.frst)\n")
    end
    for scale in (0.9, 1.0, 1.1)
        name = _scale_name(scale)
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
    (; leading_count, term_count=size(loaded.Q, 2),
       cy_volume=geometric.cy_volume)
end

function main()
    data_root = get(ENV, "CYAXIVERSE_DATA_DIR", DATA_ROOT)
    output = get(ENV, "CYAXIVERSE_AUTHOR_BRIDGE_OUTPUT", DEFAULT_OUTPUT)
    ispath(output) && throw(ArgumentError(
        "output already exists; choose a new CYAXIVERSE_AUTHOR_BRIDGE_OUTPUT: $output"))
    selected = select_geometries(data_root)
    mkpath(output)
    selection_path = joinpath(output, "selection.csv")
    open(selection_path, "w") do io
        println(io, "sample_order,h11,polytope,frst")
        for (order, geometry) in enumerate(selected)
            println(io, "$order,$(geometry.h11),$(geometry.polytope),$(geometry.frst)")
        end
    end
    for (order, geometry) in enumerate(selected)
        output_dir = joinpath(output, @sprintf("geometry_%03d", order))
        metadata = export_package_inputs(geometry, output_dir)
        println("exported $order/$(length(selected)) " *
            "geometry=$(geometry.h11),$(geometry.polytope),$(geometry.frst) " *
            "leading=$(metadata.leading_count) terms=$(metadata.term_count)")
    end
    bridge = joinpath(@__DIR__, "run_author_code_coefficient_bridge.py")
    result_path = joinpath(output, "author_results.json")
    ENV["CYAXIVERSE_DATA_DIR"] = data_root
    run(`$AUTHOR_PYTHON $bridge --input-root $output --author-src $AUTHOR_SOURCE --output-json $result_path`)
    println("selection=$selection_path results=$result_path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
