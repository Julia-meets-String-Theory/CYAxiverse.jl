# CYAX-0232 R-002: invariance artifacts for the potential and the vacua.
#
# Usage (from a checkout whose package is under test):
#     julia --startup-file=no --project=<checkout> scripts/r002_invariance_artifacts.jl <out_dir>
#
# The script writes `<out_dir>/r002_artifacts.txt`, a canonical text dump, and
# `<out_dir>/r002_sha256.txt`, one sha256 per fixture and method plus one for the
# whole dump. The R-002 comparison runs this script at the implementation base
# and at the candidate in the same CI job, so both runs share the Julia patch
# release, OS image and BLAS (spec R-002, *Bitwise comparison*).
#
# Recorded per fixture (spec R-002):
#   (a) the stored potential data `Q`, `L` and the stored `Kinv`, read back
#       through `read.potential_factored` and `read.potential`;
#   (b), (c) the `vacua_pipeline` group datasets (`threshold`, `estimate`,
#       `issquare`, `extrarows`, `verified`, and the coordinate datasets) and the
#       count fields of its search metadata, for each search method, plus the
#       direct `vacua_id` and `vacua_estimate` results;
#   (d) the ϑ-Hessian ∂²V/∂ϑ∂ϑ at each stored vacuum coordinate column, read as
#       a period-one ϑ point in the GLSM basis, with
#       V(ϑ) = Σₐ Λₐ (1 − cos(2π qₐ·ϑ)) and Λₐ = sign · 10^log10 from `L`.
#       This array does not depend on K.
#
# Run-specific metadata (timings, git revision, completion time, Julia version,
# configuration digest) is excluded. Floats are written as the hex of their
# IEEE-754 bits, so the dump compares bitwise. Errors are recorded by type and
# first message line, so a fixture that fails at the base must fail the same way.

using CYAxiverse, HDF5, LinearAlgebra, Random, SHA

const PROJECT_DIR = dirname(Base.active_project())
include(joinpath(PROJECT_DIR, "scripts", "vacua_pipeline.jl"))

const SEED = 20261112
const STARTS = 64
const METHODS = (:legacy, :leading_branches, :reduced_jlm, :auto)
const METADATA_ALLOWLIST = ("status", "terminal_status", "solver_status",
    "estimate_status", "verification_status", "search_method",
    "search_classification", "legacy_search_classification", "minimum_count",
    "multiplicity", "critical_count", "branch_count", "det_Qtilde",
    "auto_selected_method", "search_status", "model_scope",
    "full_potential_status")

# Fixtures: (name, h11, np index, Kinv, Q (h11 × N), L (2 × N: sign; log10 |Λ|)).
const FIXTURES = (
    (name="r004_nondiagonal", h11=2, np=1,
     Kinv=Float64[3 1; 1 2], Q=Int[1 1 0; 0 1 1], L=Float64[1 1 1; 0 -0.5 -1.0]),
    (name="c3_h11_3", h11=3, np=2,
     Kinv=Float64[3 1 0; 1 2 0.5; 0 0.5 1.5],
     Q=Int[1 0 0 1; 0 1 0 1; 0 0 1 1], L=Float64[1 1 1 1; 0 -0.4 -0.9 -1.5]),
    (name="geometry_2_2_1", h11=2, np=3,
     Kinv=Float64[2 0; 0 1], Q=Int[1 0 1; 0 1 1], L=Float64[1 1 1; 0 -1 -10]),
    (name="non_unimodular_qtilde", h11=2, np=4,
     Kinv=Float64[2 0.5; 0.5 1], Q=Int[2 0 1; 0 1 1], L=Float64[1 1 1; 0 -1 -10]),
    (name="ill_conditioned_kinv", h11=2, np=5,
     Kinv=Float64[1.0 1.0 - 5e-11; 1.0 - 5e-11 1.0],
     Q=Int[1 0 1; 0 1 1], L=Float64[1 1 1; -20 -21 -22]),
    (name="merged_subleading_charge", h11=2, np=6,
     Kinv=Float64[3 1; 1 2], Q=Int[1 1 0; 0 1 1], L=Float64[1 1 1; 0 -0.5 -0.7]),
    (name="h11_3_det_6_relevant_subleading", h11=3, np=7,
     Kinv=Float64[3 1 0; 1 2 0.5; 0 0.5 1.5],
     Q=Int[2 0 0 1; 0 3 0 1; 0 0 1 1], L=Float64[1 1 1 1; 0 -0.3 -0.6 -0.8]),
)

hexbits(x::Float64) = string(reinterpret(UInt64, x); base=16, pad=16)

canon(x::Float64) = "f:" * hexbits(x)
canon(x::Integer) = "i:" * string(x)
canon(x::Rational) = "q:" * string(numerator(x)) * "/" * string(denominator(x))
canon(x::AbstractFloat) = canon(Float64(x))
canon(x::AbstractString) = "s:" * repr(String(x))
canon(x::Symbol) = "y:" * string(x)
canon(x::Bool) = "b:" * string(x)
canon(::Nothing) = "nothing"
function canon(x::AbstractArray)
    "a" * string(size(x)) * "[" * join((canon(v) for v in vec(collect(x))), ",") * "]"
end
canon(x::Tuple) = "t(" * join((canon(v) for v in x), ",") * ")"
function canon(x::NamedTuple)
    "n{" * join((string(k) * "=" * canon(getfield(x, k)) for k in sort(collect(keys(x)))), ",") * "}"
end
function canon(x::AbstractDict)
    "d{" * join((string(k) * "=" * canon(x[k]) for k in sort(collect(keys(x)); by=string)), ",") * "}"
end
canon(x) = "o:" * string(typeof(x)) * ":" * repr(x)

errline(err) = "error:" * string(nameof(typeof(err))) * ":" *
    first(split(sprint(showerror, err), '\n'))

function write_geometry(root, f)
    dir = joinpath(root, "h11_" * lpad(f.h11, 3, '0'), "np_" * lpad(f.np, 7, '0'), "cy_0000001")
    mkpath(dir)
    h5open(joinpath(dir, "cyax.h5"), "w") do file
        cytools = create_group(file, "cytools")
        potential = create_group(cytools, "potential")
        potential["Q"] = f.Q
        potential["L"] = f.L
        geometric = create_group(cytools, "geometric")
        geometric["Kinv"] = f.Kinv
    end
    joinpath(dir, "cyax.h5")
end

function theta_hessian(Q::AbstractMatrix{<:Integer}, L::AbstractMatrix{<:Real}, ϑ::AbstractVector{Float64})
    n, N = size(Q)
    H = zeros(Float64, n, n)
    for a in 1:N
        Λ = Float64(L[1, a]) * 10.0^Float64(L[2, a])
        q = Float64.(Q[:, a])
        c = (2π)^2 * Λ * cos(2π * dot(q, ϑ))
        for i in 1:n, j in 1:n
            H[i, j] += c * q[i] * q[j]
        end
    end
    H
end

function pipeline_records(path)
    lines = String[]
    h5open(path, "r") do file
        haskey(file, "vacua_pipeline") || return push!(lines, "vacua_pipeline=absent")
        group = file["vacua_pipeline"]
        for name in ("threshold", "estimate", "issquare", "extrarows", "verified")
            haskey(group, name) && push!(lines, "vacua_pipeline/$name=" * canon(read(group[name])))
        end
        for coords in ("theta_min", "theta_parallel")
            haskey(group, coords) || continue
            num = read(group[coords]["numerator"])
            den = read(group[coords]["denominator"])
            push!(lines, "vacua_pipeline/$coords/numerator=" * canon(num))
            push!(lines, "vacua_pipeline/$coords/denominator=" * canon(den))
        end
        if haskey(group, "metadata")
            metadata = group["metadata"]
            for name in METADATA_ALLOWLIST
                haskey(metadata, name) && push!(lines, "vacua_pipeline/metadata/$name=" * canon(read(metadata[name])))
            end
        end
    end
    lines
end

function stored_coordinates(path)
    columns = Vector{Float64}[]
    h5open(path, "r") do file
        haskey(file, "vacua_pipeline") || return
        group = file["vacua_pipeline"]
        for coords in ("theta_min", "theta_parallel")
            haskey(group, coords) || continue
            num = read(group[coords]["numerator"])
            den = read(group[coords]["denominator"])
            values = num ./ den
            values isa AbstractMatrix || (values = reshape(values, :, 1))
            for col in eachcol(values)
                push!(columns, Float64.(col))
            end
        end
    end
    columns
end

function fixture_records(root, f)
    lines = String["fixture=" * f.name]
    path = write_geometry(root, f)
    geom_idx = CYAxiverse.structs.GeometryIndex(f.h11, f.np, 1)
    ENV["CYAXIVERSE_DATA_DIR"] = root

    # (a) stored potential data and Kinv, read back through the package readers.
    try
        factored = CYAxiverse.read.potential_factored(geom_idx)
        push!(lines, "potential_factored/Q=" * canon(factored.Q))
        push!(lines, "potential_factored/L=" * canon(factored.L))
        push!(lines, "potential_factored/Kinv=" * canon(factored.Kinv))
    catch err
        push!(lines, "potential_factored=" * errline(err))
    end
    try
        potential = CYAxiverse.read.potential(geom_idx)
        push!(lines, "potential/Q=" * canon(potential.Q))
        push!(lines, "potential/L=" * canon(potential.L))
    catch err
        push!(lines, "potential=" * errline(err))
    end

    # (b), (c) direct legacy results.
    Random.seed!(SEED)
    try
        push!(lines, "vacua_estimate=" * canon(CYAxiverse.generate.vacua_estimate(geom_idx; threshold=0.5)))
    catch err
        push!(lines, "vacua_estimate=" * errline(err))
    end
    Random.seed!(SEED)
    try
        push!(lines, "vacua_id=" * canon(CYAxiverse.generate.vacua_id(f.L, f.Q; threshold=0.5, runs=STARTS)))
    catch err
        push!(lines, "vacua_id=" * errline(err))
    end

    # (b), (c), (d) per search method, through the production pipeline.
    for method in METHODS
        push!(lines, "method=" * string(method))
        Random.seed!(SEED)
        try
            compute_vacua_data(geom_idx, root; threshold=0.5, save=true, force=true,
                starts=STARTS, method=method)
            append!(lines, pipeline_records(path))
            for (k, ϑ) in enumerate(stored_coordinates(path))
                push!(lines, "theta_hessian[$k]=" * canon(theta_hessian(f.Q, f.L, ϑ)))
            end
        catch err
            push!(lines, "compute_vacua_data=" * errline(err))
        end
    end
    lines
end

function main(out_dir)
    mkpath(out_dir)
    sections = Pair{String, Vector{String}}[]
    mktempdir() do root
        for f in FIXTURES
            push!(sections, f.name => fixture_records(root, f))
        end
    end
    dump = join((join(lines, "\n") for (_, lines) in sections), "\n\n") * "\n"
    write(joinpath(out_dir, "r002_artifacts.txt"), dump)
    open(joinpath(out_dir, "r002_sha256.txt"), "w") do io
        for (name, lines) in sections
            println(io, bytes2hex(sha256(join(lines, "\n"))), "  ", name)
        end
        println(io, bytes2hex(sha256(dump)), "  r002_artifacts.txt")
    end
    print(read(joinpath(out_dir, "r002_sha256.txt"), String))
end

isempty(ARGS) && error("usage: r002_invariance_artifacts.jl <out_dir>")
main(ARGS[1])
