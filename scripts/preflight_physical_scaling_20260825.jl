#!/usr/bin/env julia

"""Fail-closed preflight for the fixed 18 physical-scaling sidecars.

Only stored/reference-domain evidence is checked here.  No scale point,
inflation trajectory, orientifold, geometry generation, or population scan is
performed.  A physical calculation is authorized only when all 18 sidecars
and their independently replayed reference-domain checks pass.
"""

using HDF5
using LinearAlgebra
using SHA
using Printf

const WORKTREE = "/Users/vmehta/Documents/CYAxiverse/cyaxiverse/CYAxiverse.jl.worktrees/physical-scale-inflation-20260825"
const DATA_ROOT = "/Users/vmehta/Documents/CYAxiverse/cyaxiverse/data"
const MANIFEST = "/private/tmp/cyax-inflation-physical-scale-pilot-20260825/selection_manifest.json"
const MANIFEST_SHA256 = "a6df5dca258c11724d4162477cdee7cc34e5802f2f3f296a7ea64b55f23c3247"
const REQUIRED_COMMIT = "9f31d716eaab8d63d3f76826a40de5ae38c7015d"
const REQUIRED_BRANCH = "agents/physical-scale-inflation-20260825"
const CERT_DIR = joinpath(WORKTREE, "validation/physical_scaling_certificates_20260825")
const POLICY = joinpath(CERT_DIR, "physical_scaling_pilot_policy-v1.json")
const OUTPUT_JSONL = joinpath(CERT_DIR, "physical_scaling_preflight_20260825.jsonl")
const OUTPUT_MD = joinpath(CERT_DIR, "physical_scaling_preflight_20260825.md")
const OUTPUT_SHA = joinpath(CERT_DIR, "physical_scaling_preflight_20260825.sha256")
const SPD_TOLERANCE = 1e-12
const QL_TOLERANCE = 1e-10

json_escape(s::AbstractString) = begin
    io = IOBuffer()
    for c in s
        if c == '"'; print(io, "\\\"")
        elseif c == '\\'; print(io, "\\\\")
        elseif c == '\n'; print(io, "\\n")
        elseif c == '\r'; print(io, "\\r")
        elseif c == '\t'; print(io, "\\t")
        elseif UInt32(c) < 0x20; @printf(io, "\\u%04x", UInt32(c))
        else; print(io, c)
        end
    end
    String(take!(io))
end

json_value(::Nothing) = "null"
json_value(x::Missing) = "null"
json_value(x::Bool) = x ? "true" : "false"
json_value(x::Integer) = string(x)
json_value(x::AbstractFloat) = isfinite(x) ? repr(x) : string('"', json_escape(string(x)), '"')
json_value(x::Symbol) = string('"', json_escape(string(x)), '"')
json_value(x::AbstractString) = string('"', json_escape(x), '"')
json_value(x::AbstractArray) = string('[', join(json_value.(collect(x)), ','), ']')
json_value(x::Tuple) = json_value(collect(x))
function json_value(x::AbstractDict)
    keys_sorted = sort!(String[string(k) for k in keys(x)])
    string('{', join((string(json_value(key), ':', json_value(x[key])) for key in keys_sorted), ','), '}')
end

sha256_bytes(bytes::Vector{UInt8}) = bytes2hex(sha256(bytes))
sha256_file(path::AbstractString) = sha256_bytes(read(path))

function atomic_write(path::AbstractString, text::AbstractString)
    mkpath(dirname(path))
    temporary = string(path, ".tmp-", getpid(), "-", time_ns())
    try
        open(temporary, "w") do io
            write(io, text)
            flush(io)
        end
        mv(temporary, path; force=true)
    finally
        isfile(temporary) && rm(temporary; force=true)
    end
end

function metadata_string(raw::String, key::String)
    m = match(Regex("\\\"" * key * "\\\":\\\"([^\\\"]*)\\\""), raw)
    m === nothing ? nothing : m.captures[1]
end

function canonical_array_string(values)
    io = IOBuffer()
    print(io, "dtype=", string(eltype(values)), ";shape=", join(size(values), "x"), ";values=")
    for value in vec(values)
        print(io, repr(value), ",")
    end
    String(take!(io))
end

canonical_array_sha256(values) = sha256_bytes(Vector{UInt8}(codeunits(canonical_array_string(values))))

function expected_potential(Q::Matrix{Int}, tau::Vector{Float64}, kinv::Matrix{Float64}, volume::Float64)
    h11, nterms = size(Q)
    root = isqrt(8 * nterms + 1)
    root^2 == 8 * nterms + 1 || error("non-triangular charge count")
    nq = (root - 1) ÷ 2
    nq * (nq + 1) ÷ 2 == nterms || error("non-base-plus-pairwise charge count")
    output = zeros(Float64, 2, nterms)
    prefactor = 8 * π / volume^2
    log10e = log10(exp(1.0))
    for i in 1:nq
        q = @view Q[:, i]
        exponent = -2 * π * log10e * dot(q, tau)
        coefficient = prefactor * dot(q, tau)
        output[1, i] = sign(coefficient)
        output[2, i] = log10(abs(coefficient)) + exponent
    end
    index = nq + 1
    for i in 1:(nq - 1), j in (i + 1):nq
        qi, qj = @view(Q[:, i]), @view(Q[:, j])
        sumq = qi .+ qj
        exponent = -2 * π * log10e * dot(sumq, tau)
        coefficient = prefactor * (π * dot(qi, kinv * qj) + dot(sumq, tau))
        output[1, index] = sign(coefficient)
        output[2, index] = log10(abs(coefficient)) + exponent
        index += 1
    end
    output
end

function parse_sidecars()
    validator = joinpath(WORKTREE, "scripts/validate_physical_scaling_sidecars_20260825.py")
    output = read(`python3 $validator $CERT_DIR $POLICY $MANIFEST $DATA_ROOT $WORKTREE`, String)
    lines = split(chomp(output), '\n'; keepempty=false)
    length(lines) == 19 && startswith(lines[1], "META\t") || error("sidecar validator did not return 18 entries")
    rows = Dict{String,Any}[]
    for line in lines[2:end]
        fields = split(line, '\t'; keepempty=true)
        length(fields) == 15 && fields[1] == "ENTRY" || error("invalid sidecar validator entry")
        push!(rows, Dict{String,Any}(
            "name" => fields[2], "relative_path" => fields[3],
            "artifact_sha256" => fields[4], "h11" => parse(Int, fields[5]),
            "polytope" => parse(Int, fields[6]), "polytope_id" => fields[7],
            "triangulation_id" => fields[8], "sidecar_sha256" => fields[9],
            "scaling_status" => fields[10], "control_status" => fields[11],
            "ql_error" => parse(Float64, fields[12]),
            "ql_sign_mismatches" => parse(Int, fields[13]),
            "basis_sha256" => fields[14], "basis_matrix_sha256" => fields[15]))
    end
    length(rows) == 18 || error("expected 18 sidecars")
    rows
end

function one_geometry(row)
    relative = row["relative_path"]
    path = joinpath(DATA_ROOT, relative)
    sha256_file(path) == row["artifact_sha256"] || error("artifact hash mismatch: $relative")
    h5open(path, "r") do file
        raw_metadata = String(read(attributes(file)["construction_metadata_json"]))
        read_data(name) = haskey(file, name) ? read(file[name]) : error("missing dataset $name")
        h11 = row["h11"]
        tip = Float64.(read_data("cytools/geometric/tip"))
        tau = Float64.(read_data("cytools/geometric/divisor_volumes"))
        prime = Float64.(read_data("cytools/geometric/prime_divisor_volumes"))
        curves = Float64.(read_data("cytools/geometric/curve_volumes"))
        qprime = read_data("cytools/geometric/effective_cone")
        hyperplanes = Float64.(read_data("cytools/geometric/kahler_hyperplanes"))
        kinv_raw = Float64.(read_data("cytools/geometric/Kinv"))
        volume = Float64(read_data("cytools/geometric/CY_volume"))
        Q_raw = read_data("cytools/potential/Q")
        L = Float64.(read_data("cytools/potential/L"))
        basis = Int.(read_data("cytools/geometric/basis"))
        basis_matrix_stored = Int.(read_data("cytools/geometric/basis_matrix"))
        Q = Int.(round.(Q_raw)); all(Q_raw .== Q) || error("Q is not integer-valued: $relative")
        E = size(qprime, 2) == h11 ? Matrix{Float64}(qprime) :
            Matrix{Float64}(qprime')
        H = size(hyperplanes, 2) == length(tip) ? Matrix{Float64}(hyperplanes) :
            Matrix{Float64}(hyperplanes')
        effective = E * tau
        margin_values = H * tip
        margin = minimum(margin_values)
        direct_ok = size(Q, 1) == h11 && size(Q, 2) >= size(E, 1) && Q[:, 1:size(E,1)] == Int.(round.(E'))
        pair_ok = direct_ok
        index = size(E, 1) + 1
        if pair_ok
            for i in 1:(size(E,1) - 1), j in (i + 1):size(E,1)
                pair_ok &= Q[:, index] == Q[:, j] - Q[:, i]
                index += 1
            end
        end
        kinv_sym = Matrix(Symmetric(kinv_raw))
        K = Matrix(inv(Symmetric(kinv_raw)))
        identity = Matrix{Float64}(I, h11, h11)
        raw_asym = maximum(abs.(kinv_raw - kinv_raw'))
        K_asym = maximum(abs.(K - K'))
        inverse_residual = maximum(abs.(K * kinv_sym - identity))
        shifted_K = try cholesky(Symmetric(K - SPD_TOLERANCE * identity)); true catch; false end
        shifted_Kinv = try cholesky(Symmetric(kinv_sym - SPD_TOLERANCE * identity)); true catch; false end
        expected_L = expected_potential(Q, tau, kinv_sym, volume)
        ql_error = maximum(abs.(L[2,:] .- expected_L[2,:]))
        ql_sign_mismatches = count(i -> sign(L[1,i]) != sign(expected_L[1,i]), axes(L,2))
        ql_status = ql_sign_mismatches == 0 && ql_error <= QL_TOLERANCE ? "pass_existing_tolerance" : (ql_sign_mismatches == 0 ? "fail_log10_error_threshold" : "fail_sign_mismatch")
        ql_status == "pass_existing_tolerance" || error("Q/L diagnostic failed cited tolerance: $relative")
        row["scaling_status"] == "passed" || error("sidecar scaling status changed: $relative")
        row["ql_sign_mismatches"] == ql_sign_mismatches || error("sidecar Q/L sign count mismatch: $relative")
        abs(row["ql_error"] - ql_error) <= max(1e-15, eps(ql_error)) || error("sidecar Q/L error mismatch: $relative")
        direct_ok && pair_ok || error("charge orientation failed: $relative")
        all(isfinite, tip) && all(isfinite, tau) && all(isfinite, prime) && all(isfinite, curves) && all(isfinite, effective) && all(isfinite, margin_values) && all(isfinite, kinv_raw) && all(isfinite, K) && all(isfinite, L) || error("non-finite evidence: $relative")
        volume > 0 && minimum(prime) > 0 && minimum(effective) > 0 && minimum(curves) > 0 && margin > 0 || error("positive-volume/Kahler check failed: $relative")
        raw_asym <= SPD_TOLERANCE && K_asym <= SPD_TOLERANCE && inverse_residual <= SPD_TOLERANCE && shifted_K && shifted_Kinv || error("symmetry/inverse/SPD check failed: $relative")
        basis_matrix = size(basis_matrix_stored, 1) == h11 ? basis_matrix_stored : (size(basis_matrix_stored, 2) == h11 ? basis_matrix_stored' : error("basis matrix orientation failed"))
        canonical_array_sha256(basis) == row["basis_sha256"] || error("basis identity hash mismatch: $relative")
        canonical_array_sha256(basis_matrix) == row["basis_matrix_sha256"] || error("basis matrix identity hash mismatch: $relative")
        Dict{String,Any}(
            "relative_path" => relative, "h11" => row["h11"],
            "polytope" => row["polytope"], "polytope_id" => row["polytope_id"],
            "triangulation_id" => row["triangulation_id"],
            "artifact_sha256" => row["artifact_sha256"],
            "sidecar_sha256" => row["sidecar_sha256"],
            "physical_scaling_gate_status" => "passed",
            "physical_scaling_gate_reason" => "independent reference-domain replay passed under approved diagnostic policy",
            "physical_control_gate_status" => row["control_status"],
            "physical_control_gate_reason" => "control evidence remains independent and not_established where absent",
            "physical_viability_status" => "not_evaluated",
            "stored_QL_status" => ql_status,
            "stored_QL_max_log10_error" => ql_error,
            "stored_QL_sign_mismatches" => ql_sign_mismatches,
            "stored_QL_threshold" => QL_TOLERANCE,
            "stored_QL_threshold_status" => "passed_existing_cited_tolerance",
            "raw_Kinv_asymmetry_max_abs" => raw_asym,
            "K_symmetry_max_abs" => K_asym,
            "inverse_residual_max_abs" => inverse_residual,
            "K_eigmin" => eigmin(Symmetric(K)),
            "Kinv_eigmin" => eigmin(Symmetric(kinv_sym)),
            "shifted_K_cholesky" => shifted_K,
            "shifted_Kinv_cholesky" => shifted_Kinv,
            "minimum_prime_divisor_volume" => minimum(prime),
            "minimum_effective_divisor_volume" => minimum(effective),
            "minimum_curve_volume" => minimum(curves),
            "kahler_margin" => margin,
            "scale_calculation_started" => false)
    end
end

function main()
    started = time()
    isfile(POLICY) || error("physical-scaling policy is missing")
    isfile(MANIFEST) || error("fixed selection manifest is missing")
    sha256_file(MANIFEST) == MANIFEST_SHA256 || error("fixed selection manifest hash changed")
    rows = parse_sidecars()
    records = Dict{String,Any}[]
    failure = nothing
    for row in rows
        try
            push!(records, one_geometry(row))
        catch error
            failure = sprint(showerror, error)
            push!(records, Dict{String,Any}(
                "relative_path" => row["relative_path"], "h11" => row["h11"],
                "polytope" => row["polytope"], "physical_scaling_gate_status" => "failed",
                "physical_scaling_gate_reason" => failure,
                "physical_control_gate_status" => row["control_status"],
                "physical_control_gate_reason" => "preflight failed before control qualification",
                "physical_viability_status" => "blocked_scaling_gate",
                "scale_calculation_started" => false))
        end
    end
    passed = count(record -> record["physical_scaling_gate_status"] == "passed", records)
    status = passed == 18 && failure === nothing ? "passed" : "failed"
    summary = Dict{String,Any}(
        "schema" => "physical-scaling-preflight-v1", "status" => status,
        "geometry_count" => 18, "physical_scaling_gate_passed" => passed,
        "physical_scaling_gate_required" => 18,
        "physical_control_gate_not_established" => count(record -> get(record, "physical_control_gate_status", "") == "not_established", records),
        "scale_calculation_authorized" => status == "passed",
        "scale_calculation_started" => false, "inflation_evaluated" => false,
        "orientifold_computed" => false, "geometry_generated" => false,
        "population_expanded" => false, "database_written" => false,
        "failure" => failure, "records" => records,
        "resource_limits" => Dict("maximum_resident_bytes" => 2000000000,
            "maximum_new_output_bytes" => 2000000000,
            "one_geometry_at_a_time" => true),
        "elapsed_seconds" => time() - started, "maxrss_bytes" => Sys.maxrss())
    jsonl = json_value(summary) * "\n"
    atomic_write(OUTPUT_JSONL, jsonl)
    jsonl_sha = sha256_file(OUTPUT_JSONL)
    atomic_write(OUTPUT_SHA, string(jsonl_sha, "  ", basename(OUTPUT_JSONL), "\n"))
    report = IOBuffer()
    println(report, "# Physical scaling preflight — 2026-08-25")
    println(report)
    println(report, "Status: `", status, "`.")
    println(report)
    println(report, "- Exact fixed inputs: 18; physical scaling gates passed: ", passed, "/18.")
    println(report, "- Scale calculation authorized: `", status == "passed", "`; scale calculation started: `false`.")
    println(report, "- Physical control gate is independent; `not_established` is permitted for this diagnostic pilot and blocks viability claims.")
    println(report, "- Q/L status is explicit for every record with the cited tolerance `1e-10`.")
    println(report, "- Raw Kinv asymmetry was measured before symmetrization; the declared absolute SPD tolerance is `1e-12`.")
    println(report, "- Preflight JSONL SHA-256: `", jsonl_sha, "`; max RSS: `", summary["maxrss_bytes"], "` bytes; elapsed: `", summary["elapsed_seconds"], "` seconds.")
    atomic_write(OUTPUT_MD, String(take!(report)))
    println("preflight_status=", status)
    println("physical_scaling_gate_passed=", passed, "/18")
    println("scale_calculation_authorized=", status == "passed")
    println("preflight_jsonl_sha256=", sha256_file(OUTPUT_JSONL))
    println("preflight_report_sha256=", sha256_file(OUTPUT_MD))
    println("preflight_checksum_sha256=", sha256_file(OUTPUT_SHA))
    println("maxrss_bytes=", summary["maxrss_bytes"])
    println("new_output_bytes=", filesize(OUTPUT_JSONL) + filesize(OUTPUT_MD) + filesize(OUTPUT_SHA))
    status == "passed" || exit(1)
end

main()
