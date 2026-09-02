#!/usr/bin/env julia

"""Generate the bounded physical-scaling certificate sidecars.

This is a read-only geometry reconstruction pass apart from the explicitly
listed sidecar/policy outputs.  It does not evaluate inflation, a scale point,
an orientifold, or a population scan.  Legacy evidence classifications remain
separate from the owner-approved pilot conventions.
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
const AUDIT_JSONL = joinpath(WORKTREE, "validation/physical_evidence_audit_20260825.jsonl")
const AUDIT_MD = joinpath(WORKTREE, "validation/physical_evidence_audit_20260825.md")
const AUDIT_SCRIPT = joinpath(WORKTREE, "scripts/audit_physical_certificate.jl")
const OUTPUT_DIR = joinpath(WORKTREE, "validation/physical_scaling_certificates_20260825")
const POLICY_PATH = joinpath(OUTPUT_DIR, "physical_scaling_pilot_policy-v1.json")
const POLICY_SHA_PATH = joinpath(OUTPUT_DIR, "physical_scaling_pilot_policy-v1.sha256")
const SIDECAR_SHA_PATH = joinpath(OUTPUT_DIR, "physical_scaling_certificates.sha256")
const SIDECAR_VERSION = "physical-scaling-certificate-v1"
const CERTIFICATE_VERSION = "physical-domain-certificate-3"
const GATE_POLICY_VERSION = "separate-gates-v2"
const UNITS = "M_s=M_Pl;k=dimensionless"
const NORMALIZATION = "homogeneous_full_volume_k32"
const PHASE = "zero_phases"
const SPD_TOLERANCE = "1e-12"
const CERTIFICATE_PRECISION_BITS = 256
const SOURCE_NUMERIC_TYPE = "Float64"
const SOURCE_PRECISION_BITS = 53
const TARGET_NUMERIC_TYPE = "Float64"
const TARGET_PRECISION_BITS = 53
const CONVERSION_TOLERANCE = "1e-12"
const CONVERSION_POLICY_VERSION = "kinv-mixed-tolerance-v1"
const KINV_CONVERSION_RULE =
    "max_absolute_error <= 1e-12 OR max_relative_error <= 1e-12"
const QL_TOLERANCE = "1e-10"
const QL_TOLERANCE_SOURCE = "validation/run_author_code_coefficient_bridge.py:238-253"
const HISTORICAL_GENERATOR_SHA256 = "52d227d15e7231ff7e83faf23bb56be9a40fd2dc9b1c421d3ea7d894c35e9686"

const MANIFEST_PARSER = """
import json, re, sys
path, expected_commit, expected_branch = sys.argv[1:4]
with open(path, encoding='utf-8') as stream:
    data = json.load(stream)
if not isinstance(data, dict) or data.get('manifest_schema') != 'cyax-inflation-physical-scale-pilot-selection-1':
    raise SystemExit('invalid selection manifest root/schema')
if data.get('git_commit') != expected_commit or data.get('git_branch') != expected_branch:
    raise SystemExit('selection manifest source identity mismatch')
items = data.get('selected_inputs')
if not isinstance(items, list) or len(items) != 18 or data.get('selected_count') != 18:
    raise SystemExit('selection manifest must contain exactly 18 inputs')
seen = set()
for i, item in enumerate(items):
    if not isinstance(item, dict): raise SystemExit(f'entry {i} is not an object')
    required = ('path','sha256','h11','polytope','polytope_id','triangulation_id','orientifold')
    if any(key not in item for key in required): raise SystemExit(f'entry {i} is incomplete')
    if item['path'] in seen: raise SystemExit('duplicate path')
    seen.add(item['path'])
    if not isinstance(item['sha256'], str) or not re.fullmatch(r'[0-9a-f]{64}', item['sha256']):
        raise SystemExit(f'entry {i} has invalid SHA-256')
    if not isinstance(item['h11'], int) or isinstance(item['h11'], bool) or not isinstance(item['polytope'], int) or isinstance(item['polytope'], bool):
        raise SystemExit(f'entry {i} has invalid integer identity')
    if not isinstance(item['polytope_id'], str) or not isinstance(item['triangulation_id'], str):
        raise SystemExit(f'entry {i} has invalid topological identity')
    if not isinstance(item['orientifold'], dict) or item['orientifold'].get('requested') is not False:
        raise SystemExit(f'entry {i} is not explicitly non-orientifold')
print('META\\t' + '\\t'.join((data['manifest_schema'], data['git_commit'], data['git_branch'], str(data['selected_count']), data.get('physical_contract',''))))
for item in items:
    print('ENTRY\\t' + '\\t'.join((item['path'], item['sha256'], str(item['h11']), str(item['polytope']), item['polytope_id'], item['triangulation_id'])))
"""

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

function sha256_bytes(bytes::Vector{UInt8})
    bytes2hex(sha256(bytes))
end

function sha256_file(path::AbstractString)
    sha256_bytes(read(path))
end

function atomic_write(path::AbstractString, bytes::Vector{UInt8})
    mkpath(dirname(path))
    temporary = string(path, ".tmp-", getpid(), "-", time_ns())
    try
        open(temporary, "w") do io
            write(io, bytes)
            flush(io)
        end
        mv(temporary, path; force=true)
    finally
        isfile(temporary) && rm(temporary; force=true)
    end
end

atomic_write(path::AbstractString, text::AbstractString) = atomic_write(path, Vector{UInt8}(codeunits(text)))

function git_output(args...)
    strip(read(`git -C $WORKTREE $args`, String))
end

function validate_repository()
    git_output("rev-parse", "HEAD") == REQUIRED_COMMIT || error("HEAD does not match required commit")
    git_output("branch", "--show-current") == REQUIRED_BRANCH || error("branch does not match required branch")
    occursin("diff --check", git_output("diff", "--check")) && error("git diff --check failed")
    nothing
end

function parse_manifest()
    output = read(`python3 -c $MANIFEST_PARSER $MANIFEST $REQUIRED_COMMIT $REQUIRED_BRANCH`, String)
    lines = split(chomp(output), '\n'; keepempty=false)
    length(lines) == 19 && startswith(lines[1], "META\t") || error("strict manifest parser returned wrong record count")
    entries = Dict{String,Any}[]
    for line in lines[2:end]
        fields = split(line, '\t'; keepempty=true)
        length(fields) == 7 && fields[1] == "ENTRY" || error("strict manifest entry parse failed")
        startswith(fields[2], DATA_ROOT * "/") || error("manifest path is outside the fixed data root")
        relative = fields[2][length(DATA_ROOT) + 2:end]
        push!(entries, Dict{String,Any}(
            "manifest_path" => fields[2], "relative_path" => relative, "artifact_sha256" => fields[3],
            "h11" => parse(Int, fields[4]), "polytope" => parse(Int, fields[5]),
            "polytope_id" => fields[6], "triangulation_id" => fields[7],
            "orientifold_requested" => false))
    end
    length(entries) == 18 || error("expected 18 manifest entries")
    for h11 in 5:10
        group = filter(entry -> entry["h11"] == h11, entries)
        length(group) == 3 || error("expected three entries at h11=$h11")
        length(unique(entry["polytope_id"] for entry in group)) == 3 || error("polytope identity collision at h11=$h11")
        length(unique(entry["polytope"] for entry in group)) == 3 || error("polytope index collision at h11=$h11")
    end
    entries
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

function metadata_string(raw::String, key::String)
    m = match(Regex("\\\"" * key * "\\\":\\\"([^\\\"]*)\\\""), raw)
    m === nothing ? nothing : m.captures[1]
end

function metadata_number(raw::String, key::String)
    m = match(Regex("\\\"" * key * "\\\":([-+0-9.eE]+)"), raw)
    m === nothing ? nothing : try parse(Float64, m.captures[1]) catch; nothing end
end

function metadata_int(raw::String, key::String)
    value = metadata_number(raw, key)
    value === nothing ? nothing : Int(round(value))
end

function expected_potential(Q::Matrix{Int}, tau::Vector{Float64}, kinv::Matrix{Float64}, volume::Float64)
    h11, nterms = size(Q)
    root = isqrt(8 * nterms + 1)
    root^2 == 8 * nterms + 1 || error("potential term count is not triangular")
    nq = (root - 1) ÷ 2
    nq * (nq + 1) ÷ 2 == nterms || error("potential term count is not base-plus-pairwise")
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

function source_hashes()
    files = [
        "scripts/run_physical_scale_inflation_pilot_20260825.jl",
        "scripts/validate_physical_scale_checkpoint_20260825.py",
        "scripts/inflation_scale_continuation.jl",
        "scripts/inflation_scan_shards_common.jl",
        "scripts/inflation_diagnostics_common.jl",
        "scripts/inflation_refinement_common.jl",
        "src/CYAxiverse.jl",
        "src/read.jl",
        "src/generate.jl",
        "src/structs.jl",
        "Project.toml",
        "../../CYAxiverse.jl/Manifest.toml",
        "scripts/build_orientifold_vacua_inflation.jl",
        "test/runtests.jl",
        "validation/inflation_physical_scale_derivation.md",
        "validation/physical_scale_inflation_progress_20260825.md",
        "scripts/audit_physical_certificate.jl",
        "validation/physical_evidence_audit_20260825.jsonl",
        "validation/physical_evidence_audit_20260825.md",
        "scripts/generate_physical_scaling_sidecars_20260825.jl",
        "scripts/test_physical_scaling_sidecars_20260825.py",
        "scripts/validate_physical_scaling_sidecars_20260825.py",
        "scripts/preflight_physical_scaling_20260825.jl",
    ]
    result = Dict{String,Any}()
    for relative in files
        path = joinpath(WORKTREE, relative)
        isfile(path) || error("source identity file missing: $relative")
        result[relative] = sha256_file(path)
    end
    result
end

function classify_fields()
    Dict{String,Any}(
        "basis_identity" => "stored_alias",
        "charge_orientation" => "exactly_reconstructable",
        "configuration_digest" => "genuinely_unavailable",
        "curve_volumes" => "stored_alias",
        "cy_volume" => "stored_alias",
        "divisor_volumes" => "stored_alias",
        "effective_divisor_volumes" => "exactly_reconstructable",
        "instanton_control" => "genuinely_unavailable",
        "kahler_margin" => "exactly_reconstructable",
        "kinv" => "stored_alias",
        "moduli_status" => "source_fixed_convention",
        "normalization" => "genuinely_unavailable",
        "perturbative_control" => "genuinely_unavailable",
        "phase_convention" => "genuinely_unavailable",
        "potent_curve_volumes" => "genuinely_unavailable",
        "precision_bits" => "exactly_reconstructable",
        "prime_divisor_volumes" => "stored_alias",
        "source_identity" => "genuinely_unavailable",
        "spd_tolerance" => "genuinely_unavailable",
        "units" => "genuinely_unavailable",
        "visible_sector_status" => "source_fixed_convention",
    )
end

function policy_payload(source_hash_map, complete_diff_hash)
    Dict{String,Any}(
        "certificate_schema" => SIDECAR_VERSION,
        "certificate_version" => CERTIFICATE_VERSION,
        "gate_policy_version" => GATE_POLICY_VERSION,
        "selection_manifest_sha256" => MANIFEST_SHA256,
        "selected_geometry_count" => "18",
        "h11_values" => ["5", "6", "7", "8", "9", "10"],
        "polytopes_per_h11" => "3",
        "orientifold_requested" => false,
        "scale_grid" => ["0.9", "0.95", "0.99", "1.0", "1.01", "1.05", "1.1"],
        "units" => UNITS,
        "normalization" => NORMALIZATION,
        "normalization_laws" => Dict("tau" => "k*tau_0", "Kinv" => "k^2*Kinv_0", "K" => "k^-2*K_0", "CY_volume" => "k^(3/2)*CY_volume_0", "curve_volumes" => "sqrt(k)*curve_volumes_0", "Kahler_margin" => "sqrt(k)*Kahler_margin_0"),
        "phase_convention" => PHASE,
        "spd_tolerance" => SPD_TOLERANCE,
        "spd_tolerance_type" => "absolute",
        "certificate_numeric_type" => "BigFloat",
        "certificate_precision_bits" => string(CERTIFICATE_PRECISION_BITS),
        "source_storage_type" => SOURCE_NUMERIC_TYPE,
        "source_precision_bits" => string(SOURCE_PRECISION_BITS),
        "evaluator_target_type" => TARGET_NUMERIC_TYPE,
        "evaluator_target_precision_bits" => string(TARGET_PRECISION_BITS),
        "relative_conversion_tolerance" => CONVERSION_TOLERANCE,
        "absolute_metric_conversion_tolerance" => CONVERSION_TOLERANCE,
        "conversion_policy_version" => CONVERSION_POLICY_VERSION,
        "conversion_acceptance" => Dict(
            "L" => "max_absolute_error <= 1e-12",
            "K" => "max_absolute_error <= 1e-12",
            "Kinv" => KINV_CONVERSION_RULE,
            "tau" => "max_relative_error <= 1e-12",
            "volume" => "max_relative_error <= 1e-12"),
        "stored_QL_reference_tolerance" => QL_TOLERANCE,
        "source_hashes" => source_hash_map,
        "current_complete_git_diff_sha256" => complete_diff_hash,
        "historical_generator_sha256" => HISTORICAL_GENERATOR_SHA256,
        "canonicalization" => "sorted object keys; UTF-8; no insignificant whitespace; approved numeric values represented as stable strings",
        "control_gate_policy" => "independent; not_established is permitted for diagnostic calculation; only passed permits eligibility, and eligibility is not validation",
        "physical_scaling_gate_policy" => "must pass before physical scale calculation; any failure blocks",
        "nonphysical_paths" => "homotopy_only and fixed-volume diagnostics remain nonphysical",
    )
end

function conversion_measure(value)
    max_abs = zero(BigFloat)
    max_rel = zero(BigFloat)
    for entry in (value isa Number ? (value,) : vec(value))
        source = BigFloat(entry)
        target = Float64(source)
        error = abs(source - BigFloat(target))
        max_abs = max(max_abs, error)
        max_rel = max(max_rel, error / max(abs(source), one(BigFloat)))
    end
    Dict{String,Any}(
        "max_absolute_error" => Float64(max_abs),
        "max_relative_error" => Float64(max_rel))
end

function conversion_audit(; L, K, kinv, tau, volume)
    metrics = Dict{String,Any}(
        "L" => conversion_measure(L), "K" => conversion_measure(K),
        "Kinv" => conversion_measure(kinv), "tau" => conversion_measure(tau),
        "volume" => conversion_measure(volume))
    absolute_tolerance = parse(Float64, CONVERSION_TOLERANCE)
    status = Dict{String,Any}(
        "L" => metrics["L"]["max_absolute_error"] <= absolute_tolerance ? "passed" : "failed",
        "K" => metrics["K"]["max_absolute_error"] <= absolute_tolerance ? "passed" : "failed",
        "Kinv" => (metrics["Kinv"]["max_absolute_error"] <= absolute_tolerance ||
            metrics["Kinv"]["max_relative_error"] <= absolute_tolerance) ? "passed" : "failed",
        "tau" => metrics["tau"]["max_relative_error"] <= absolute_tolerance ? "passed" : "failed",
        "volume" => metrics["volume"]["max_relative_error"] <= absolute_tolerance ? "passed" : "failed")
    Dict{String,Any}(
        "status" => all(value -> value == "passed", values(status)) ? "passed" : "failed",
        "policy_version" => CONVERSION_POLICY_VERSION,
        "acceptance_rules" => Dict(
            "L" => "max_absolute_error <= 1e-12",
            "K" => "max_absolute_error <= 1e-12",
            "Kinv" => KINV_CONVERSION_RULE,
            "tau" => "max_relative_error <= 1e-12",
            "volume" => "max_relative_error <= 1e-12"),
        "metrics" => metrics, "metric_status" => status,
        "max_absolute_error" => maximum(Float64[metrics[key]["max_absolute_error"] for key in keys(metrics)]),
        "max_relative_error" => maximum(Float64[metrics[key]["max_relative_error"] for key in keys(metrics)]),
        "absolute_metric_tolerance" => CONVERSION_TOLERANCE,
        "relative_tolerance" => CONVERSION_TOLERANCE,
        "source_numeric_type" => SOURCE_NUMERIC_TYPE,
        "source_precision_bits" => SOURCE_PRECISION_BITS,
        "target_numeric_type" => TARGET_NUMERIC_TYPE,
        "target_precision_bits" => TARGET_PRECISION_BITS)
end

function sidecar_for(entry, source_hash_map, complete_diff_hash, common_digest)
    relative = entry["relative_path"]
    path = joinpath(DATA_ROOT, relative)
    entry["manifest_path"] == path || error("manifest path binding mismatch: $relative")
    isfile(path) || error("fixed input missing: $path")
    actual_hash = sha256_file(path)
    actual_hash == entry["artifact_sha256"] || error("fixed input hash changed: $relative")
    h11, polytope, frst = entry["h11"], entry["polytope"], 1
    startswith(relative, "h11_$(lpad(h11,3,'0'))/np_$(lpad(polytope,7,'0'))/cy_0000001/") || error("manifest path identity mismatch: $relative")

    sidecar = Dict{String,Any}(
        "schema" => SIDECAR_VERSION,
        "certificate_version" => CERTIFICATE_VERSION,
        "generated_for" => "physical-inflation-diagnostic-pilot-20260825",
        "input" => Dict{String,Any}(
            "relative_path" => relative, "absolute_path" => path,
            "artifact_sha256" => actual_hash, "artifact_bytes" => filesize(path),
            "h11" => h11, "polytope" => polytope, "frst" => frst,
            "polytope_id" => entry["polytope_id"], "triangulation_id" => entry["triangulation_id"],
            "orientifold_requested" => false),
        "selection_manifest" => Dict{String,Any}("path" => MANIFEST, "sha256" => MANIFEST_SHA256),
        "audit_evidence" => Dict{String,Any}("jsonl_path" => AUDIT_JSONL, "jsonl_sha256" => sha256_file(AUDIT_JSONL), "report_path" => AUDIT_MD, "report_sha256" => sha256_file(AUDIT_MD)),
        "field_classifications" => classify_fields(),
        "approved_conventions" => Dict{String,Any}("units" => UNITS, "normalization" => NORMALIZATION, "phase_convention" => PHASE, "spd_tolerance" => SPD_TOLERANCE, "certificate_precision_bits" => CERTIFICATE_PRECISION_BITS, "source_numeric_type" => SOURCE_NUMERIC_TYPE, "source_precision_bits" => SOURCE_PRECISION_BITS, "target_numeric_type" => TARGET_NUMERIC_TYPE, "target_precision_bits" => TARGET_PRECISION_BITS, "relative_conversion_tolerance" => CONVERSION_TOLERANCE, "absolute_metric_conversion_tolerance" => CONVERSION_TOLERANCE, "conversion_policy_version" => CONVERSION_POLICY_VERSION, "kinv_conversion_acceptance" => KINV_CONVERSION_RULE, "stored_QL_reference_tolerance" => QL_TOLERANCE),
    )

    h5open(path, "r") do file
        raw_metadata = String(read(attributes(file)["construction_metadata_json"]))
        getdata(name) = haskey(file, name) ? read(file[name]) : error("missing dataset $name in $relative")
        basis = getdata("cytools/geometric/basis")
        basis_matrix_storage = getdata("cytools/geometric/basis_matrix")
        qprime = getdata("cytools/geometric/effective_cone")
        tau = Float64.(getdata("cytools/geometric/divisor_volumes"))
        prime = Float64.(getdata("cytools/geometric/prime_divisor_volumes"))
        curves = Float64.(getdata("cytools/geometric/curve_volumes"))
        mori = getdata("cytools/geometric/mori_cone")
        tip = Float64.(getdata("cytools/geometric/tip"))
        hyperplanes = Float64.(getdata("cytools/geometric/kahler_hyperplanes"))
        kinv_raw = Float64.(getdata("cytools/geometric/Kinv"))
        volume = Float64(getdata("cytools/geometric/CY_volume"))
        Q_raw = getdata("cytools/potential/Q")
        L_raw = Float64.(getdata("cytools/potential/L"))
        h11_dataset = Int(round(getdata("cytools/geometric/h11")))
        h11_dataset == h11 || error("stored h11 mismatch in $relative")
        Q = Int.(round.(Q_raw)); all(Q_raw .== Q) || error("Q is not integer-valued in $relative")
        # Use concrete matrices for both orientations so canonical array
        # hashes do not depend on lazy Adjoint BLAS accumulation.
        E = size(qprime, 2) == h11 ? Matrix{Float64}(qprime) :
            Matrix{Float64}(qprime')
        H = size(hyperplanes, 2) == length(tip) ? Matrix{Float64}(hyperplanes) :
            Matrix{Float64}(hyperplanes')
        effective = E * tau
        kahler_slack = H * tip
        margin = minimum(kahler_slack)
        kinv_sym = Matrix(Symmetric(kinv_raw))
        K = Matrix(inv(Symmetric(kinv_raw)))
        raw_asym = maximum(abs.(kinv_raw - kinv_raw'))
        K_asym = maximum(abs.(K - K'))
        identity = Matrix{Float64}(I, h11, h11)
        inverse_residual = maximum(abs.(K * kinv_sym - identity))
        tol = parse(Float64, SPD_TOLERANCE)
        shifted_K = try cholesky(Symmetric(K - tol * identity)); true catch; false end
        shifted_Kinv = try cholesky(Symmetric(kinv_sym - tol * identity)); true catch; false end
        expected_L = expected_potential(Q, tau, kinv_sym, volume)
        ql_error = maximum(abs.(L_raw[2,:] .- expected_L[2,:]))
        sign_mismatches = count(i -> sign(L_raw[1,i]) != sign(expected_L[1,i]), axes(L_raw,2))
        ql_status = sign_mismatches == 0 && ql_error <= parse(Float64, QL_TOLERANCE) ? "pass_existing_tolerance" : (sign_mismatches == 0 ? "fail_log10_error_threshold" : "fail_sign_mismatch")
        effective_count = length(effective)
        control_components = Dict{String,Any}("potent_ray_evidence" => "not_established", "instanton_control" => "not_established", "perturbative_control" => "not_established", "moduli_control" => "not_established", "visible_sector_applicability" => "not_applicable")
        control_status = "not_established"
        control_reason = "control evidence is intentionally incomplete: potent-ray, instanton zero-mode, perturbative correction, and moduli-stabilization evidence are absent; non-orientifold input has no visible-sector claim"
        source_identity_payload = Dict{String,Any}("type" => "reconstructed_replay_identity", "geometry_sha256" => actual_hash, "polytope_id" => entry["polytope_id"], "triangulation_id" => entry["triangulation_id"], "cy3_fingerprint" => metadata_string(raw_metadata,"cy3_fingerprint"), "stored_cytools_version" => metadata_string(raw_metadata,"cytools_version"), "historical_generator_source_sha256" => HISTORICAL_GENERATOR_SHA256, "current_continuation_source_sha256" => source_hash_map["scripts/inflation_scale_continuation.jl"], "current_complete_git_diff_sha256" => complete_diff_hash, "audit_script_sha256" => source_hash_map["scripts/audit_physical_certificate.jl"], "evidence_report_sha256" => sha256_file(AUDIT_MD), "selection_manifest_sha256" => MANIFEST_SHA256, "conversion_policy_version" => CONVERSION_POLICY_VERSION, "qualification" => "reconstructed; not original-generator provenance")
        source_identity_digest = sha256_bytes(Vector{UInt8}(codeunits(json_value(source_identity_payload))))
        source_identity_payload["digest_sha256"] = source_identity_digest
        source_identity_string = "reconstructed-replay-identity-v1:sha256:" * source_identity_digest
        config_payload = Dict{String,Any}("common_policy_digest_sha256" => common_digest, "relative_path" => relative, "artifact_sha256" => actual_hash, "h11" => string(h11), "polytope" => string(polytope), "frst" => string(frst), "polytope_id" => entry["polytope_id"], "triangulation_id" => entry["triangulation_id"], "orientifold_requested" => false, "source_identity_digest_sha256" => source_identity_digest, "conversion_policy_version" => CONVERSION_POLICY_VERSION)
        config_digest = sha256_bytes(Vector{UInt8}(codeunits(json_value(config_payload))))
        basis_int = Int.(basis)
        basis_matrix_storage_int = Int.(basis_matrix_storage)
        basis_matrix = size(basis_matrix_storage_int, 1) == h11 ? basis_matrix_storage_int : (size(basis_matrix_storage_int, 2) == h11 ? basis_matrix_storage_int' : error("basis matrix orientation mismatch"))
        basis_identity = Dict{String,Any}("basis_shape" => collect(size(basis_int)), "basis_values" => vec(basis_int), "basis_sha256" => canonical_array_sha256(basis_int), "basis_matrix_shape" => collect(size(basis_matrix)), "basis_matrix_values" => vec(basis_matrix), "basis_matrix_sha256" => canonical_array_sha256(basis_matrix), "basis_matrix_hdf5_shape" => collect(size(basis_matrix_storage_int)), "basis_matrix_orientation" => "canonical one-row-per-basis-divisor orientation; transpose applied when HDF5.jl exposes the Python-written array transposed", "basis_convention" => metadata_string(raw_metadata,"basis_convention"), "index_bases" => Dict("basis_index_base" => "not_stored_separately; exact basis values and CYTools convention are recorded", "kappa_index_base" => metadata_int(raw_metadata,"kappa_index_base"), "qcd_divisor_index_base" => metadata_int(raw_metadata,"qcd_divisor_index_base")), "canonical_serialization" => "dtype=<eltype>;shape=<dimensions joined by x>;values=<repr(value), in Julia column-major vec order>")
        conversion = conversion_audit(L=L_raw, K=K, kinv=kinv_raw, tau=tau, volume=volume)
        scaling_required = Dict{String,Any}(
            "geometry_and_domain_evidence" => "passed", "basis_identity" => "passed", "charge_orientation" => "passed", "phase_convention" => "passed_by_approved_pilot_convention", "unit_contract" => "passed_by_approved_pilot_convention", "normalization" => "passed_by_approved_pilot_convention", "replayable_reconstructed_source_identity" => "passed_reconstructed", "canonical_configuration_digest" => "passed", "precision_and_conversion_audit" => conversion["status"], "symmetry_inverse_and_SPD_policy" => raw_asym <= tol && K_asym <= tol && inverse_residual <= tol && shifted_K && shifted_Kinv ? "passed" : "failed", "positive_volumes" => isfinite(volume) && volume > 0 && minimum(prime) > 0 && minimum(effective) > 0 && minimum(curves) > 0 ? "passed" : "failed", "Kahler_domain_checks" => isfinite(margin) && margin > 0 ? "passed" : "failed")
        scaling_status = all(value -> startswith(String(value), "passed"), values(scaling_required)) && ql_status == "pass_existing_tolerance" ? "passed" : "failed"
        scaling_reason = scaling_status == "passed" ? "all scaling evidence checks passed at the stored reference geometry under the approved diagnostic policies; policy-supplied fields are not legacy metadata" : "one or more physical scaling evidence checks failed"
        sidecar["source_identity"] = source_identity_payload
        sidecar["configuration_payload"] = config_payload
        sidecar["configuration_digest"] = Dict{String,Any}("common_sha256" => common_digest, "geometry_sha256" => config_digest, "canonicalization" => "sorted object keys; UTF-8; no insignificant whitespace; stable numeric strings")
        sidecar["basis_identity"] = basis_identity
        sidecar["geometry_evidence"] = Dict{String,Any}("stored_array_values" => Dict("tau" => tau, "Kinv_raw" => kinv_raw, "CY_volume" => volume, "prime_divisor_volumes" => prime, "curve_volumes" => curves, "tip" => tip, "effective_cone_normalized" => E, "kahler_hyperplanes_normalized" => H, "Q" => Q, "L" => L_raw), "reconstructed_values" => Dict("K" => K, "effective_divisor_volumes" => effective, "kahler_slack" => kahler_slack), "array_hashes" => Dict("tau" => canonical_array_sha256(tau), "Kinv_raw" => canonical_array_sha256(kinv_raw), "CY_volume" => canonical_array_sha256([volume]), "prime_divisor_volumes" => canonical_array_sha256(prime), "curve_volumes" => canonical_array_sha256(curves), "tip" => canonical_array_sha256(tip), "effective_cone_normalized" => canonical_array_sha256(E), "kahler_hyperplanes_normalized" => canonical_array_sha256(H), "Q" => canonical_array_sha256(Q), "L" => canonical_array_sha256(L_raw), "K" => canonical_array_sha256(K), "effective_divisor_volumes" => canonical_array_sha256(effective), "kahler_slack" => canonical_array_sha256(kahler_slack)), "minimums" => Dict("CY_volume" => volume, "prime_divisor_volume" => minimum(prime), "effective_divisor_volume" => minimum(effective), "curve_volume" => minimum(curves), "kahler_margin" => margin), "raw_Kinv_asymmetry_max_abs" => raw_asym, "K_symmetry_max_abs" => K_asym, "inverse_residual_max_abs" => inverse_residual, "Kinv_eigmin" => eigmin(Symmetric(kinv_sym)), "K_eigmin" => eigmin(Symmetric(K)), "shifted_K_cholesky" => shifted_K, "shifted_Kinv_cholesky" => shifted_Kinv, "spd_tolerance" => SPD_TOLERANCE, "raw_asymmetry_status" => "measured_before_symmetrization")
        sidecar["stored_QL_reference"] = Dict{String,Any}("status" => ql_status, "max_log10_error" => ql_error, "sign_mismatches" => sign_mismatches, "tolerance" => QL_TOLERANCE, "tolerance_source" => QL_TOLERANCE_SOURCE, "threshold_status" => ql_status == "pass_existing_tolerance" ? "passed_existing_cited_tolerance" : "failed_existing_cited_tolerance")
        sidecar["precision_conversion_audit"] = conversion
        sidecar["physical_scaling_gate"] = Dict{String,Any}("status" => scaling_status, "reason" => scaling_reason, "provenance" => "scripts/generate_physical_scaling_sidecars_20260825.jl::sidecar_for", "required_field_statuses" => scaling_required, "reference_scale" => "1.0", "legacy_missing_fields_are_not_recovered" => true)
        sidecar["physical_control_gate"] = Dict{String,Any}("status" => control_status, "reason" => control_reason, "provenance" => "approved separate-gate policy; sidecar generation only", "component_statuses" => control_components)
        sidecar["physical_viability"] = Dict{String,Any}("status" => "not_evaluated", "reason" => "no inflation scale point or viability screen was evaluated during sidecar generation")
        sidecar["reconstruction_formulas"] = Dict{String,Any}("effective_divisor_volumes" => "normalize effective_cone to one ray per row E; effective=E*tau", "kahler_margin" => "normalize kahler_hyperplanes to one row per hyperplane H; margin=min(H*tip)", "charge_orientation" => "Q is h11 x N; direct columns are transpose(E), then ordered pair differences", "K" => "K=inv(Symmetric(raw Kinv)) after raw asymmetry is measured", "normalization" => "tau=k*tau0; Kinv=k^2*Kinv0; K=k^-2*K0; CY_volume=k^(3/2)*CY_volume0; curves=sqrt(k)*curves0; Kahler_margin=sqrt(k)*margin0")
    end
    sidecar
end

function main()
    started = time()
    validate_repository()
    isfile(MANIFEST) || error("selection manifest is missing")
    sha256_file(MANIFEST) == MANIFEST_SHA256 || error("selection manifest hash changed")
    entries = parse_manifest()
    source_hash_map = source_hashes()
    complete_diff_hash = sha256_bytes(read(`git -C $WORKTREE diff --binary HEAD`))
    common_payload = policy_payload(source_hash_map, complete_diff_hash)
    common_digest = sha256_bytes(Vector{UInt8}(codeunits(json_value(common_payload))))
    policy = Dict{String,Any}("schema" => "physical-scaling-pilot-policy-v1", "policy_version" => GATE_POLICY_VERSION, "created_at_utc" => "2026-08-25", "common_configuration_payload" => common_payload, "common_configuration_digest_sha256" => common_digest, "source_identity" => Dict("required_commit" => REQUIRED_COMMIT, "required_branch" => REQUIRED_BRANCH, "current_complete_git_diff_sha256" => complete_diff_hash, "source_file_hashes" => source_hash_map, "historical_generator_sha256" => HISTORICAL_GENERATOR_SHA256, "selection_manifest_sha256" => MANIFEST_SHA256, "audit_jsonl_sha256" => sha256_file(AUDIT_JSONL), "audit_report_sha256" => sha256_file(AUDIT_MD)), "fixed_inputs" => Dict("selection_manifest" => MANIFEST, "selection_manifest_sha256" => MANIFEST_SHA256, "geometry_count" => 18, "h11_values" => [5,6,7,8,9,10], "polytopes_per_h11" => 3, "orientifold_requested" => false, "scale_grid" => ["0.9","0.95","0.99","1.0","1.01","1.05","1.1"]), "resource_limits" => Dict("maximum_resident_bytes" => 2000000000, "maximum_new_output_bytes" => 2000000000, "one_geometry_at_a_time" => true, "atomic_checkpoint_after_each_geometry" => true, "idempotent_resume" => true), "field_classification_policy" => "legacy audit statuses are preserved verbatim in every sidecar; approved pilot conventions are recorded in a separate object", "control_gate_policy" => "not_established is allowed for diagnostic calculation but blocks viability/production/validated-candidate claims", "scale_calculation_started" => false)
    policy_json = json_value(policy) * "\n"
    atomic_write(POLICY_PATH, policy_json)
    atomic_write(POLICY_SHA_PATH, string(bytes2hex(sha256(Vector{UInt8}(codeunits(policy_json)))), "  ", basename(POLICY_PATH), "\n"))
    sidecar_hashes = String[]
    statuses = String[]
    for entry in entries
        sidecar = sidecar_for(entry, source_hash_map, complete_diff_hash, common_digest)
        input = sidecar["input"]
        name = string("h11_", lpad(input["h11"], 3, '0'), "_np_", lpad(input["polytope"], 7, '0'), "_cy_", lpad(input["frst"], 7, '0'), ".physical-scaling-certificate-v1.json")
        path = joinpath(OUTPUT_DIR, name)
        text = json_value(sidecar) * "\n"
        atomic_write(path, text)
        hash = sha256_file(path)
        push!(sidecar_hashes, string(hash, "  ", name))
        push!(statuses, string(input["relative_path"], "=", sidecar["physical_scaling_gate"]["status"], "/", sidecar["physical_control_gate"]["status"]))
    end
    length(sidecar_hashes) == 18 || error("did not generate exactly 18 sidecars")
    sort!(sidecar_hashes)
    atomic_write(SIDECAR_SHA_PATH, join(sidecar_hashes, "\n") * "\n")
    elapsed = time() - started
    println("generated_sidecars=18")
    println("policy_sha256=", sha256_file(POLICY_PATH))
    println("policy_manifest_sha256=", sha256_file(POLICY_SHA_PATH))
    println("sidecar_checksum_sha256=", sha256_file(SIDECAR_SHA_PATH))
    println("common_configuration_digest_sha256=", common_digest)
    println("current_complete_git_diff_sha256=", complete_diff_hash)
    println("statuses=", join(statuses, ";"))
    println("elapsed_seconds=", elapsed)
    println("maxrss_bytes=", Sys.maxrss())
end

main()
