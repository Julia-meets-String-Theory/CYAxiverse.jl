# CYAX-0232 R-005: c = 1/2 comparison harness (T013) and count report.
#
# Usage (from a checkout whose package is under test):
#     julia --startup-file=no --project=<checkout> scripts/r005_scaling_harness.jl dump <out_dir> <kinv_scale>
#     julia --startup-file=no --project=<checkout> scripts/r005_scaling_harness.jl report <out_dir>
#
# `dump` writes `<out_dir>/r005_routes.txt`, a canonical dump of every R-005
# route output (floats as IEEE-754 hex), and its sha256. T013 verification:
# `dump` at the implementation base (c = 1/2 code) with kinv_scale 1 equals
# `dump` at the candidate (c = 1 code) with kinv_scale 2, byte for byte. Run
# both from this file, with `--project` selecting the package.
#
# `report` evaluates the candidate at kinv_scale 1 (c = 1) and 2 (c = 1/2) and
# writes `<out_dir>/r005_report.md`: the measured ratio for every route output
# and every count or classification that changed (R-005 "reported" rows).
#
# The routes are in scripts/r005_scaling_routes.jl. The script functions they
# call are loaded from the checkout under test.

using CYAxiverse, HDF5, LinearAlgebra, Printf, SHA

const R005_PROJECT_DIR = dirname(Base.active_project())
include(joinpath(R005_PROJECT_DIR, "scripts", "batch_physical_spectrum.jl"))
include(joinpath(R005_PROJECT_DIR, "scripts", "build_orientifold_vacua_inflation.jl"))
include(joinpath(@__DIR__, "r005_scaling_routes.jl"))

r005_hex(x::Float64) = "f:" * string(reinterpret(UInt64, x); base=16, pad=16)
r005_canon(x::Float64) = r005_hex(x)
r005_canon(x::AbstractFloat) = r005_hex(Float64(x))
r005_canon(x::Integer) = "i:" * string(x)
r005_canon(x::Bool) = "b:" * string(x)
r005_canon(x::AbstractString) = "s:" * repr(String(x))
r005_canon(x::AbstractArray) =
    "a" * string(size(x)) * "[" * join((r005_canon(v) for v in vec(collect(x))), ",") * "]"
r005_canon(x) = "o:" * string(typeof(x)) * ":" * repr(x)

function r005_dump(out_dir, kinv_scale)
    mkpath(out_dir)
    outputs = r005_route_outputs(; kinv_scale)
    text = join((k * "=" * r005_canon(outputs[k]) for k in sort(collect(keys(outputs)))), "\n") * "\n"
    write(joinpath(out_dir, "r005_routes.txt"), text)
    digest = bytes2hex(sha256(text))
    write(joinpath(out_dir, "r005_sha256.txt"), digest * "  r005_routes.txt\n")
    println(digest, "  r005_routes.txt  (", length(outputs), " outputs, kinv_scale=", kinv_scale, ")")
end

# Log10-valued outputs: persisted `log10` datasets and the in-memory
# spectrum fields, which are stored as log10 values.
const R005_LOG_FIELDS = ("m", "f", "fK", "fpert", "λself", "λ31", "λ22",
    "mass_min", "mass_max", "fK_log10", "fpert_log10")
function r005_is_log_field(key)
    name = last(split(key, '/'))
    occursin("log10", name) || name in R005_LOG_FIELDS ||
        occursin("/decay/", key)
end

function r005_report(out_dir)
    mkpath(out_dir)
    c1 = r005_route_outputs(; kinv_scale=1)
    chalf = r005_route_outputs(; kinv_scale=2)
    lines = String["# CYAX-0232 R-005 report", "",
        "Candidate evaluated at c = 1 (stored Kinv) and c = 1/2 (2·Kinv).",
        "Log10 fields: offset c1 − c½. Linear fields: ratio c1/c½ (max |deviation| over entries).", "",
        "| Output | Kind | c = 1 vs c = 1/2 |", "|---|---|---|"]
    for key in sort(collect(keys(c1)))
        a, b = c1[key], get(chalf, key, missing)
        b === missing && (push!(lines, "| `$key` | — | missing at c = 1/2 |"); continue)
        if a isa AbstractString || eltype(a) <: AbstractString
            push!(lines, "| `$key` | string | " * (a == b ? "equal" : "differs") * " |")
        elseif eltype(a) <: Integer
            push!(lines, "| `$key` | count | " * (a == b ? "equal: $(a)" : "changed: $(b) → $(a)") * " |")
        elseif size(a) != size(b)
            push!(lines, "| `$key` | float | shape changed: $(size(b)) → $(size(a)) |")
        elseif r005_is_log_field(key)
            d = Float64.(a) .- Float64.(b)
            finite = isfinite.(d)
            summary = any(finite) ?
                @sprintf("offset %.12f (spread %.1e)", first(d[finite]),
                    maximum(d[finite]) - minimum(d[finite])) : "no finite entries"
            push!(lines, "| `$key` | log10 | $summary |")
        else
            r = Float64.(a) ./ Float64.(b)
            finite = isfinite.(r)
            summary = any(finite) ?
                @sprintf("ratio %.12f (spread %.1e)", first(r[finite]),
                    maximum(r[finite]) - minimum(r[finite])) : "no finite entries"
            push!(lines, "| `$key` | linear | $summary |")
        end
    end
    write(joinpath(out_dir, "r005_report.md"), join(lines, "\n") * "\n")
    println(join(lines, "\n"))
end

if abspath(PROGRAM_FILE) == @__FILE__
    mode = isempty(ARGS) ? "" : ARGS[1]
    if mode == "dump" && length(ARGS) == 3
        r005_dump(ARGS[2], parse(Float64, ARGS[3]))
    elseif mode == "report" && length(ARGS) == 2
        r005_report(ARGS[2])
    else
        error("usage: r005_scaling_harness.jl dump <out_dir> <kinv_scale> | report <out_dir>")
    end
end
