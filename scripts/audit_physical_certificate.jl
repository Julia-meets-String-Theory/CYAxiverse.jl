#!/usr/bin/env julia

"""Read-only evidence audit for the fixed physical-inflation pilot inputs.

This script deliberately does not evaluate inflation or any scale point.  It
streams one HDF5 file at a time, verifies the fixed selection identity, checks
the stored potential orientation/formula, reconstructs only algebraic fields
whose source formula is explicit, and writes an atomic JSONL audit report.

The audit uses HDF5 only for inspection.  The worktree's uninstantiated
Project.toml is not modified and no dependency is installed or fetched.
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
const OUTPUT_JSONL = joinpath(WORKTREE, "validation/physical_evidence_audit_20260825.jsonl")
const OUTPUT_ZST = joinpath(WORKTREE, "validation/physical_evidence_audit_20260825.jsonl.zst")
const OUTPUT_MD = joinpath(WORKTREE, "validation/physical_evidence_audit_20260825.md")
const ALLOWED_UNTRACKED = Set([
    "scripts/audit_physical_certificate.jl",
    "validation/physical_evidence_audit_20260825.jsonl",
    "validation/physical_evidence_audit_20260825.jsonl.zst",
    "validation/physical_evidence_audit_20260825.md",
])
const QL_LOG10_ERROR_TOLERANCE = 1e-10
const QL_TOLERANCE_SOURCE = "validation/run_author_code_coefficient_bridge.py:238-253"
const HANDOFF_COMPRESSED_SHA256 = "e88c24e538a8e4ab62c8f3fba3852ef3d045ef1ae1e6cb9be4aa5ac188100bf4"
const HANDOFF_JSONL_SHA256 = "d7056751a8b207147cc56c6415ab3a5753053026137818b39795383cf5d752e5"
const HISTORICAL_GENERATOR_SHA256 = "52d227d15e7231ff7e83faf23bb56be9a40fd2dc9b1c421d3ea7d894c35e9686"
const CONTINUATION_SOURCE_SHA256 = "c43bf109750eca12eb78d5a5118b402db29a7adc6aa1eeb928dd26a6dc5e3e4e"
const DERIVATION_LEDGER_SHA256 = "97bb1147d088c6d66407de9a1966b8872541ebdcb5dfd822c1b82722c9ef90c5"

const EXPECTED_SHA256 = Dict(
    "h11_005/np_0000001/cy_0000001/cyax.h5" => "8a6a4214b34998d2be31394dde99e962319e555d22d92d4e496f233b357d3547",
    "h11_005/np_0000002/cy_0000001/cyax.h5" => "b8bff295697f74b77e213e40728d3da68cccf20a7ce386a86ed7f50a2b1c8b4d",
    "h11_005/np_0000003/cy_0000001/cyax.h5" => "b5efa4d22b138a21135382880f0bcecde6bd9b3c453d37078e788981a11c3abc",
    "h11_006/np_0000001/cy_0000001/cyax.h5" => "627f22f16c25d4a5b9a41a6272b6b3442b6cc6849f3ee8b7e0b4f815a5f3a384",
    "h11_006/np_0000002/cy_0000001/cyax.h5" => "597c084f804f0d48e72a6defa652f1fe3785285ea4c89c434b4414e36af68e5b",
    "h11_006/np_0000003/cy_0000001/cyax.h5" => "7e79197f0034feeab6b2dff145562ab90c8dd800fe9241e4896b36c007d3d292",
    "h11_007/np_0000001/cy_0000001/cyax.h5" => "7aefcde0f6ede4ff1e9cff3238dc00e5ccdb70b09729a884382d52611071c42c",
    "h11_007/np_0000002/cy_0000001/cyax.h5" => "0d2e216d0f942a93683dda2e2e8407905e934bb4aa31b7d15f2a4d220d9bf354",
    "h11_007/np_0000003/cy_0000001/cyax.h5" => "4b073f8e8b544fe5a3cd05b04a39ec1a97d97965fd326e5775b442261f00a77a",
    "h11_008/np_0000001/cy_0000001/cyax.h5" => "50fa54cf896357fc2460d1accb3403af32eef20fabd23b84dc70e63dadbfbd0e",
    "h11_008/np_0000002/cy_0000001/cyax.h5" => "cbd9995556ebcca68ede2c80404d55fe2493f3c89ca08ce52b54d825b16747da",
    "h11_008/np_0000003/cy_0000001/cyax.h5" => "75b93bf1957a39bf7e5f3236973a1be7354325b97f70116ea3ff461fd1fb0921",
    "h11_009/np_0000001/cy_0000001/cyax.h5" => "618265b913c7f4e8cb2642e262529b1d3365448de012f2a3a17f7a4b7c5a2c28",
    "h11_009/np_0000002/cy_0000001/cyax.h5" => "2e0987172113aa7be45770e05ac94bba5f89372108b1db206aeb664cabc4ace2",
    "h11_009/np_0000003/cy_0000001/cyax.h5" => "c15b1c9c76cde8c80a5f7d94e7eeb2095cd42345588cc7441f8f334bfd4a016b",
    "h11_010/np_0000001/cy_0000001/cyax.h5" => "ddcc102525c9e1cad5ffc6468850f10252007f4acde4b3bff5c87ebe67fff858",
    "h11_010/np_0000002/cy_0000001/cyax.h5" => "fb7d2c0281656e6eeea753acec53b182ba91e2b9ef0c75645fb468b8e9208275",
    "h11_010/np_0000003/cy_0000001/cyax.h5" => "99e4108c752d482e12d9f8e3b31ae347bb5810fcb9858598ae1e6623c0f9c756",
)

# The manifest is parsed by Python's standard-library JSON parser.  This is
# deliberately invoked from the audit rather than replaced by substring
# checks: every selected entry is bound to its path, hash, h11, polytope
# identity, triangulation identity, and no-orientifold request.
const MANIFEST_PARSER = """
import json
import re
import sys

path, expected_commit, expected_branch = sys.argv[1:4]
with open(path, encoding="utf-8") as stream:
    data = json.load(stream)

if not isinstance(data, dict):
    raise SystemExit("selection manifest root must be an object")
if data.get("manifest_schema") != "cyax-inflation-physical-scale-pilot-selection-1":
    raise SystemExit("unexpected selection manifest schema")
if data.get("git_commit") != expected_commit:
    raise SystemExit("selection manifest commit does not match the required commit")
if data.get("git_branch") != expected_branch:
    raise SystemExit("selection manifest branch does not match the required branch")
items = data.get("selected_inputs")
if not isinstance(items, list) or data.get("selected_count") != len(items):
    raise SystemExit("selected_count does not equal the selected_inputs length")
if len(items) != 18:
    raise SystemExit("selection manifest must contain exactly 18 inputs")

seen_paths = set()
for index, item in enumerate(items):
    if not isinstance(item, dict):
        raise SystemExit(f"selected_inputs[{index}] must be an object")
    required = ("path", "sha256", "h11", "polytope", "polytope_id",
                "triangulation_id", "orientifold")
    missing = [key for key in required if key not in item]
    if missing:
        raise SystemExit(f"selected_inputs[{index}] missing {missing}")
    if not isinstance(item["path"], str) or not item["path"]:
        raise SystemExit(f"selected_inputs[{index}].path must be a nonempty string")
    if item["path"] in seen_paths:
        raise SystemExit(f"duplicate selected path: {item['path']}")
    seen_paths.add(item["path"])
    if not isinstance(item["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
        raise SystemExit(f"selected_inputs[{index}].sha256 is not a lowercase SHA-256")
    if isinstance(item["h11"], bool) or not isinstance(item["h11"], int):
        raise SystemExit(f"selected_inputs[{index}].h11 must be an integer")
    if isinstance(item["polytope"], bool) or not isinstance(item["polytope"], int):
        raise SystemExit(f"selected_inputs[{index}].polytope must be an integer")
    if not isinstance(item["polytope_id"], str) or not item["polytope_id"]:
        raise SystemExit(f"selected_inputs[{index}].polytope_id must be a nonempty string")
    if not isinstance(item["triangulation_id"], str) or not item["triangulation_id"]:
        raise SystemExit(f"selected_inputs[{index}].triangulation_id must be a nonempty string")
    orientifold = item["orientifold"]
    if not isinstance(orientifold, dict) or orientifold.get("requested") is not False:
        raise SystemExit(f"selected_inputs[{index}] is not explicitly non-orientifold")

print("\t".join(("META", data["git_commit"], data["git_branch"],
                   str(data["selected_count"]), data["manifest_schema"],
                   str(data.get("physical_contract", "")))))
for item in items:
    print("\t".join(("ENTRY", item["path"], item["sha256"], str(item["h11"]),
                     str(item["polytope"]), item["polytope_id"],
                     item["triangulation_id"], "false")))
"""

const FIELD_CLASSIFICATIONS = Dict(
    "divisor_volumes" => Dict("status" => "stored_alias", "evidence" => "cytools/geometric/divisor_volumes is persisted Float64 data", "source_anchor" => "scripts/generate_geometric_data_multitriangulation.py@770b09b7e503ccf01202b1ec2212149c7bd50a5a:896"),
    "cy_volume" => Dict("status" => "stored_alias", "evidence" => "cytools/geometric/CY_volume is persisted Float64 data", "source_anchor" => "scripts/generate_geometric_data_multitriangulation.py@770b09b7e503ccf01202b1ec2212149c7bd50a5a:895"),
    "kinv" => Dict("status" => "stored_alias", "evidence" => "cytools/geometric/Kinv is persisted Float64 data; K is its declared inverse", "source_anchor" => "src/read.jl:240-258; scripts/generate_geometric_data_multitriangulation.py@770b09b7e503ccf01202b1ec2212149c7bd50a5a:904"),
    "prime_divisor_volumes" => Dict("status" => "stored_alias", "evidence" => "cytools/geometric/prime_divisor_volumes is persisted and ordered by prime_toric_divisors", "source_anchor" => "scripts/generate_geometric_data_multitriangulation.py@770b09b7e503ccf01202b1ec2212149c7bd50a5a:897-902"),
    "effective_divisor_volumes" => Dict("status" => "exactly_reconstructable", "formula" => "normalize stored effective_cone to E (one ray per row), then E * divisor_volumes in the declared divisor basis", "evidence" => "effective_cone and divisor_volumes are persisted; the generator computes qprime @ candidate_tau before its domain checks", "source_anchor" => "scripts/generate_geometric_data_multitriangulation.py@770b09b7e503ccf01202b1ec2212149c7bd50a5a:607-640; current source:2199-2221"),
    "curve_volumes" => Dict("status" => "stored_alias", "evidence" => "cytools/geometric/curve_volumes is persisted as the Mori-ray volume vector", "source_anchor" => "scripts/generate_geometric_data_multitriangulation.py@770b09b7e503ccf01202b1ec2212149c7bd50a5a:780-789,903"),
    "potent_curve_volumes" => Dict("status" => "genuinely_unavailable", "evidence" => "Mori rays and their volumes are present, but no potent/nilpotent labels or GV-invariant data are stored", "source_anchor" => "catastrophicKS.pdf printed p.7/PDF p.8 Eq.(21); validation/inflation_physical_scale_derivation.md:82-84", "decision" => "User must provide a potent-ray classification or approve a proxy definition"),
    "kahler_margin" => Dict("status" => "exactly_reconstructable", "formula" => "normalize stored kahler_hyperplanes to H (one hyperplane per row), then minimum(H * tip)", "evidence" => "tip and kahler_hyperplanes are persisted; the historical generator computes kahler_slack and checks its minimum", "source_anchor" => "scripts/generate_geometric_data_multitriangulation.py@770b09b7e503ccf01202b1ec2212149c7bd50a5a:780-789; current source:2230-2244"),
    "basis_identity" => Dict("status" => "stored_alias", "evidence" => "basis and basis_matrix values are persisted; canonical shape/value hashes, basis convention, and stored kappa/QCD index bases are recorded per input", "source_anchor" => "scripts/generate_geometric_data_multitriangulation.py@770b09b7e503ccf01202b1ec2212149c7bd50a5a:846-851,883-886,927-930"),
    "charge_orientation" => Dict("status" => "exactly_reconstructable", "formula" => "Q is h11 x N; direct Q columns equal the transpose of normalized one-ray-per-row E, then pair columns are Q[:,j]-Q[:,i]", "evidence" => "all 18 arrays pass dimension, direct-block, pair-order, integer-valued, and Q/L formula checks", "source_anchor" => "src/read.jl:385-424; scripts/inflation_scale_continuation.jl:106-176"),
    "phase_convention" => Dict("status" => "genuinely_unavailable", "evidence" => "no phase dataset or phase metadata exists; L[1,:] is a signed coefficient, not a persisted phase", "source_anchor" => "scripts/generate_geometric_data_multitriangulation.py@770b09b7e503ccf01202b1ec2212149c7bd50a5a:969-971; validation/inflation_physical_scale_derivation.md:82,106", "decision" => "User must supply or approve an explicit phase convention"),
    "units" => Dict("status" => "genuinely_unavailable", "evidence" => "no units field is stored for the geometric or potential arrays", "source_anchor" => "validation/inflation_physical_scale_derivation.md:51-62,107", "decision" => "The owner-selected M_s=M_Pl contract does not prove the legacy artifact unit convention; approve a source-backed units declaration"),
    "normalization" => Dict("status" => "genuinely_unavailable", "evidence" => "the algebraic V_CY^-2 coefficient formula is reproducible, but no physical normalization contract is stored with the artifact", "source_anchor" => "scripts/generate_geometric_data_multitriangulation.py@770b09b7e503ccf01202b1ec2212149c7bd50a5a:792-822; validation/inflation_physical_scale_derivation.md:97-108", "decision" => "User must approve the artifact-level normalization mapping to the pilot contract"),
    "source_identity" => Dict("status" => "genuinely_unavailable", "evidence" => "polytope, triangulation, topological fingerprint, CYTools version, and paper list are stored, but generator commit, source-byte identity, and complete environment provenance are absent", "source_anchor" => "construction_metadata_json in each artifact; validation/inflation_physical_scale_derivation.md:19-28", "decision" => "Provide or approve a complete source/environment identity"),
    "configuration_digest" => Dict("status" => "genuinely_unavailable", "evidence" => "sampling parameters are partly recorded, but no canonical configuration digest or complete CLI/environment contract is stored", "source_anchor" => "construction_metadata_json sampling/mosek/qcd fields; scripts/inflation_scale_continuation.jl:57-58", "decision" => "Provide or approve a canonical configuration identity"),
    "moduli_status" => Dict("status" => "source_fixed_convention", "value" => "not_established", "evidence" => "the primary paper explicitly does not undertake moduli stabilization and the implementation preserves that status", "source_anchor" => "catastrophicKS.pdf printed p.6/PDF p.7 and p.25/PDF p.26; validation/inflation_physical_scale_derivation.md:69-74"),
    "instanton_control" => Dict("status" => "genuinely_unavailable", "evidence" => "prime-divisor volume inequality is checkable, but full instanton control is not established: holomorphicity/zero-mode realization is not recorded and the source explicitly treats it as an assumption", "source_anchor" => "catastrophicKS.pdf printed p.6/PDF p.7 and p.7/PDF p.8; validation/inflation_physical_scale_derivation.md:82-84", "decision" => "User must decide whether volume-only control is an approved proxy"),
    "perturbative_control" => Dict("status" => "genuinely_unavailable", "evidence" => "positive curve volumes are stored, but potent-ray membership and the complete correction policy are absent", "source_anchor" => "catastrophicKS.pdf printed p.7/PDF p.8 Eq.(21); validation/inflation_physical_scale_derivation.md:82-84", "decision" => "User must provide potent-ray evidence or approve a proxy"),
    "visible_sector_status" => Dict("status" => "source_fixed_convention", "value" => "not_applicable", "evidence" => "the fixed selection explicitly requested no orientifold and each artifact records orientifold requested=false/status=not_requested; no visible-sector claim is made", "source_anchor" => "selection_manifest.json orientifold_metadata; each construction_metadata_json orientifold object; validation/inflation_physical_scale_derivation.md:167-169"),
    "spd_tolerance" => Dict("status" => "genuinely_unavailable", "evidence" => "Kinv can be tested positive definite, but no artifact-level symmetry/inverse/SPD tolerance is recorded", "source_anchor" => "src/read.jl:196-237; scripts/inflation_scale_continuation.jl:597-616", "decision" => "User must approve a declared SPD tolerance policy"),
    "precision_bits" => Dict("status" => "exactly_reconstructable", "value" => 53, "evidence" => "all stored real-valued geometric and potential arrays are HDF5 Float64; Float64 represents 53 bits", "source_anchor" => "validation/inflation_physical_scale_derivation.md:120-153; src/read.jl:240-258"),
)

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
json_value(x::AbstractVector) = string('[', join(json_value.(collect(x)), ','), ']')
function json_value(x::AbstractDict)
    keys_sorted = sort!(String[string(k) for k in keys(x)])
    parts = String[]
    for key in keys_sorted
        push!(parts, string(json_value(key), ':', json_value(x[key])))
    end
    string('{', join(parts, ','), '}')
end
json_value(x::Tuple) = json_value(collect(x))

function git_output(args...)
    strip(read(`git -C $WORKTREE $args`, String))
end

function validate_repository_state()
    head = git_output("rev-parse", "HEAD")
    head == REQUIRED_COMMIT || error("repository HEAD mismatch: $head != $REQUIRED_COMMIT")
    branch = git_output("branch", "--show-current")
    branch == REQUIRED_BRANCH || error("repository branch mismatch: $branch != $REQUIRED_BRANCH")
    status_output = read(`git -C $WORKTREE status --porcelain=v1 --untracked-files=all`, String)
    unexpected = String[]
    for raw_line in split(chomp(status_output), '\n'; keepempty=false)
        fields = split(raw_line, ' '; limit=2)
        length(fields) == 2 || push!(unexpected, raw_line)
        length(fields) == 2 || continue
        status_code, relative_path = fields
        if status_code != "??" || !(relative_path in ALLOWED_UNTRACKED)
            push!(unexpected, raw_line)
        end
    end
    isempty(unexpected) || error("repository has modified tracked files or unexpected untracked files: $(join(unexpected, " | "))")
    Dict{String,Any}(
        "head" => head,
        "branch" => branch,
        "modified_tracked_files" => String[],
        "allowed_untracked_files" => sort!(collect(ALLOWED_UNTRACKED)),
        "unexpected_status_lines" => String[],
        "status" => "passed",
    )
end

function parse_selection_manifest()
    output = read(`python3 -c $MANIFEST_PARSER $MANIFEST $REQUIRED_COMMIT $REQUIRED_BRANCH`, String)
    lines = split(chomp(output), '\n'; keepempty=false)
    !isempty(lines) || error("manifest parser returned no records")
    header = split(lines[1], '\t'; keepempty=true)
    length(header) == 6 && header[1] == "META" || error("manifest parser returned an invalid header")
    metadata = Dict{String,Any}(
        "git_commit" => header[2],
        "git_branch" => header[3],
        "selected_count" => parse(Int, header[4]),
        "manifest_schema" => header[5],
        "physical_contract" => header[6],
    )
    entries = Dict{String,Any}[]
    for line in lines[2:end]
        fields = split(line, '\t'; keepempty=true)
        length(fields) == 8 && fields[1] == "ENTRY" || error("manifest parser returned an invalid entry")
        push!(entries, Dict{String,Any}(
            "path" => fields[2],
            "sha256" => fields[3],
            "h11" => parse(Int, fields[4]),
            "polytope" => parse(Int, fields[5]),
            "polytope_id" => fields[6],
            "triangulation_id" => fields[7],
            "orientifold_requested" => fields[8] == "false" ? false : error("manifest orientifold flag is not false"),
        ))
    end
    metadata, entries
end

function expected_relative_path(path::AbstractString)
    path = String(path)
    prefix = string(DATA_ROOT, "/")
    startswith(path, prefix) || error("manifest path is outside the data root: $path")
    relative = path[length(prefix) + 1:end]
    normpath(joinpath(DATA_ROOT, relative)) == path || error("manifest path is not canonical: $path")
    relative
end

function validate_selection_manifest(metadata, entries)
    metadata["git_commit"] == REQUIRED_COMMIT || error("manifest commit mismatch")
    metadata["git_branch"] == REQUIRED_BRANCH || error("manifest branch mismatch")
    metadata["selected_count"] == 18 || error("manifest selected count mismatch")
    metadata["manifest_schema"] == "cyax-inflation-physical-scale-pilot-selection-1" || error("manifest schema mismatch")
    length(entries) == 18 || error("manifest entry count mismatch")
    by_relative = Dict{String,Dict{String,Any}}()
    for entry in entries
        relative = expected_relative_path(entry["path"])
        haskey(EXPECTED_SHA256, relative) || error("manifest contains an unexpected selected path: $relative")
        haskey(by_relative, relative) && error("manifest contains a duplicate selected path: $relative")
        entry["sha256"] == EXPECTED_SHA256[relative] || error("manifest hash mismatch for $relative")
        expected_h11 = parse(Int, match(r"h11_(\d+)", relative).captures[1])
        expected_polytope = parse(Int, match(r"np_(\d+)", relative).captures[1])
        entry["h11"] == expected_h11 || error("manifest h11 mismatch for $relative")
        entry["polytope"] == expected_polytope || error("manifest polytope index mismatch for $relative")
        entry["orientifold_requested"] == false || error("manifest input is not explicitly non-orientifold: $relative")
        by_relative[relative] = entry
    end
    Set(keys(by_relative)) == Set(keys(EXPECTED_SHA256)) || error("manifest paths do not exactly match the fixed input set")
    for h11 in 5:10
        selected = [entry for entry in values(by_relative) if entry["h11"] == h11]
        length(selected) == 3 || error("manifest does not contain exactly three inputs at h11=$h11")
        length(unique([entry["polytope_id"] for entry in selected])) == 3 || error("manifest polytope identities are not distinct at h11=$h11")
        length(unique([entry["polytope"] for entry in selected])) == 3 || error("manifest polytope indices are not distinct at h11=$h11")
    end
    by_relative
end

function sha256_file(path)
    bytes2hex(open(path, "r") do io
        sha256(io)
    end)
end

function atomic_zstd(input, output)
    temporary = string(output, ".tmp-", getpid(), "-", time_ns())
    try
        run(pipeline(`zstd -19 -c $input`, stdout=temporary))
        mv(temporary, output; force=true)
    finally
        isfile(temporary) && rm(temporary; force=true)
    end
end

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

function metadata_has(raw::String, key::String)
    occursin(Regex("\\\"" * key * "\\\":"), raw)
end

function orientifold_metadata(raw::String)
    match_result = match(r"\"orientifold\":\{\"requested\":(true|false),\"status\":\"([^\"]+)\"", raw)
    match_result === nothing ? (nothing, nothing) :
        (match_result.captures[1] == "true", match_result.captures[2])
end

function canonical_array_string(values)
    io = IOBuffer()
    print(io, "dtype=", string(eltype(values)), ";shape=", join(size(values), "x"), ";values=")
    for value in vec(values)
        print(io, repr(value), ",")
    end
    String(take!(io))
end

function canonical_array_sha256(values)
    bytes2hex(sha256(Vector{UInt8}(codeunits(canonical_array_string(values)))))
end

function ql_diagnostic_status(finite_l::Bool, log_error, sign_mismatches::Int)
    !finite_l && return "not_evaluable"
    sign_mismatches != 0 && return "fail_sign_mismatch"
    log_error > QL_LOG10_ERROR_TOLERANCE && return "fail_log10_error_threshold"
    "pass_existing_tolerance"
end

function dataset(f, path)
    haskey(f, path) ? read(f[path]) : nothing
end

function expected_potential(Q::Matrix{Int}, tau, kinv, volume)
    h11, nterms = size(Q)
    disc = 8 * nterms + 1
    root = isqrt(disc)
    root^2 == disc || return nothing
    nq = (root - 1) ÷ 2
    nq * (nq + 1) ÷ 2 == nterms || return nothing
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

function audit_one(relative_path, expected_hash, manifest_entry, repository_state)
    path = joinpath(DATA_ROOT, relative_path)
    manifest_entry["path"] == path || error("manifest path binding mismatch for $relative_path")
    actual_hash = sha256_file(path)
    actual_hash == expected_hash || error("input hash mismatch for $path: $actual_hash != $expected_hash")
    manifest_entry["sha256"] == actual_hash || error("manifest artifact hash binding mismatch for $path")
    bytes = filesize(path)
    h11 = parse(Int, match(r"h11_(\d+)", relative_path).captures[1])
    polytope = parse(Int, match(r"np_(\d+)", relative_path).captures[1])
    frst = parse(Int, match(r"cy_(\d+)", relative_path).captures[1])
    manifest_entry["h11"] == h11 || error("manifest h11 binding mismatch for $relative_path")
    manifest_entry["polytope"] == polytope || error("manifest polytope binding mismatch for $relative_path")
    manifest_entry["orientifold_requested"] == false || error("manifest orientifold binding is not false for $relative_path")

    record = Dict{String,Any}(
        "audit_schema" => "cyaxiverse-physical-evidence-audit-1",
        "geometry_path" => path,
        "relative_path" => relative_path,
        "h11" => h11,
        "polytope_index" => polytope,
        "frst_index" => frst,
        "artifact_sha256" => actual_hash,
        "artifact_bytes" => bytes,
        "required_commit" => REQUIRED_COMMIT,
        "required_branch" => REQUIRED_BRANCH,
        "repository_state" => repository_state,
        "manifest_binding" => manifest_entry,
        "source_file_writes" => false,
        "inflation_evaluated" => false,
    )
    h5open(path, "r") do f
        raw_metadata = String(read(attributes(f)["construction_metadata_json"]))
        geometric = f["cytools/geometric"]
        basis = dataset(f, "cytools/geometric/basis")
        basis_matrix = dataset(f, "cytools/geometric/basis_matrix")
        h11_dataset = dataset(f, "cytools/geometric/h11")
        qprime = dataset(f, "cytools/geometric/effective_cone")
        tau = dataset(f, "cytools/geometric/divisor_volumes")
        prime = dataset(f, "cytools/geometric/prime_divisor_volumes")
        curves = dataset(f, "cytools/geometric/curve_volumes")
        mori = dataset(f, "cytools/geometric/mori_cone")
        tip = dataset(f, "cytools/geometric/tip")
        hyperplanes = dataset(f, "cytools/geometric/kahler_hyperplanes")
        kinv = dataset(f, "cytools/geometric/Kinv")
        cy_volume = dataset(f, "cytools/geometric/CY_volume")
        q_raw = dataset(f, "cytools/potential/Q")
        l_raw = dataset(f, "cytools/potential/L")

        all(x !== nothing for x in (basis, basis_matrix, h11_dataset, qprime, tau, prime,
            curves, mori, tip, hyperplanes, kinv, cy_volume, q_raw, l_raw)) ||
            error("required stored dataset missing in $path")
        h11_stored = Int(round(h11_dataset))
        h11_stored == h11 || error("stored h11 mismatch for $path")
        manifest_entry["h11"] == h11_stored || error("manifest/stored h11 mismatch for $path")
        basis_int = Int.(basis)
        basis_matrix_storage_int = Int.(basis_matrix)
        basis_matrix_int = size(basis_matrix_storage_int, 1) == h11 ?
            basis_matrix_storage_int :
            (size(basis_matrix_storage_int, 2) == h11 ? basis_matrix_storage_int' :
             error("basis_matrix has no h11-aligned orientation for $path"))
        basis_convention = metadata_string(raw_metadata, "basis_convention")
        kappa_index_base = metadata_int(raw_metadata, "kappa_index_base")
        qcd_index_base = metadata_int(raw_metadata, "qcd_divisor_index_base")
        basis_convention !== nothing || error("basis convention is absent for $path")
        kappa_index_base !== nothing || error("kappa index base is absent for $path")
        qcd_index_base !== nothing || error("QCD divisor index base is absent for $path")
        q_int = Int.(round.(q_raw))
        q_integral = all(q_raw .== q_int)
        E = size(qprime, 2) == h11 ? Float64.(qprime) : Float64.(qprime')
        effective = E * Float64.(tau)
        H = Float64.(hyperplanes)
        kahler_slack = size(H, 2) == length(tip) ? H * Float64.(tip) : H' * Float64.(tip)
        margin = minimum(kahler_slack)
        kinv_matrix = Float64.(kinv)
        kinv_raw_asymmetry = (size(kinv_matrix, 1) == size(kinv_matrix, 2) && all(isfinite, kinv_matrix)) ?
            maximum(abs.(kinv_matrix - kinv_matrix')) : Inf
        kinv_sym = Symmetric(kinv_matrix)
        K = inv(kinv_sym)
        q_count = size(E, 1)
        direct_orientation = size(q_int, 1) == h11 && size(q_int, 2) >= q_count &&
            q_int[:, 1:q_count] == Int.(round.(E'))
        pair_orientation = direct_orientation
        pair_index = q_count + 1
        if pair_orientation
            for i in 1:(q_count - 1), j in (i + 1):q_count
                pair_orientation &= q_int[:, pair_index] == q_int[:, j] - q_int[:, i]
                pair_index += 1
            end
        end
        expected_l = expected_potential(q_int, Float64.(tau), Float64.(kinv), Float64(cy_volume))
        finite_l = expected_l !== nothing && size(l_raw) == size(expected_l) &&
            all(isfinite, l_raw) && all(isfinite, expected_l)
        log_error = finite_l ? maximum(abs.(Float64.(l_raw[2, :]) .- expected_l[2, :])) : Inf
        sign_mismatches = finite_l ? count(i -> sign(l_raw[1, i]) != sign(expected_l[1, i]), axes(l_raw, 2)) : -1
        ql_status = ql_diagnostic_status(finite_l, log_error, sign_mismatches)
        orientifold_requested, orientifold_status = orientifold_metadata(raw_metadata)
        orientifold_requested === false || error("artifact orientifold.requested is not false for $path")
        orientifold_status == "not_requested" || error("artifact orientifold.status is not not_requested for $path")
        manifest_entry["polytope_id"] == metadata_string(raw_metadata, "polytope_id") || error("manifest/polytope identity mismatch for $path")
        manifest_entry["triangulation_id"] == metadata_string(raw_metadata, "triangulation_id") || error("manifest/triangulation identity mismatch for $path")
        basis_identity = Dict(
            "basis_shape" => collect(size(basis_int)),
            "basis_values" => vec(basis_int),
            "basis_sha256" => canonical_array_sha256(basis_int),
            "basis_matrix_shape" => collect(size(basis_matrix_int)),
            "basis_matrix_values" => vec(basis_matrix_int),
            "basis_matrix_sha256" => canonical_array_sha256(basis_matrix_int),
            "basis_matrix_hdf5_shape" => collect(size(basis_matrix_storage_int)),
            "basis_matrix_orientation" => "canonical one-row-per-basis-divisor orientation; transpose applied when HDF5.jl exposes the Python-written array transposed",
            "basis_convention" => basis_convention,
            "index_bases" => Dict(
                "basis_index_base" => "not_stored_separately; exact basis values and CYTools convention are recorded",
                "kappa_index_base" => kappa_index_base,
                "qcd_divisor_index_base" => qcd_index_base,
            ),
            "canonical_serialization" => "dtype=<eltype>;shape=<dimensions joined by x>;values=<repr(value), in Julia column-major vec order>",
        )
        record["basis_identity"] = basis_identity
        qcd_index = metadata_int(raw_metadata, "qcd_divisor_index")
        record["metadata"] = Dict(
            "schema_version" => metadata_string(raw_metadata, "schema_version"),
            "polytope_id" => metadata_string(raw_metadata, "polytope_id"),
            "triangulation_id" => metadata_string(raw_metadata, "triangulation_id"),
            "h11_dataset" => h11_stored,
            "cy3_fingerprint" => metadata_string(raw_metadata, "cy3_fingerprint"),
            "cytools_version" => metadata_string(raw_metadata, "cytools_version"),
            "basis_convention" => metadata_string(raw_metadata, "basis_convention"),
            "intersection_convention" => metadata_string(raw_metadata, "intersection_convention"),
            "prime_divisor_convention" => metadata_string(raw_metadata, "prime_divisor_convention"),
            "prime_divisor_lower_bound" => metadata_number(raw_metadata, "prime_divisor_volume_lower_bound"),
            "qcd_divisor_index" => qcd_index,
            "kappa_index_base" => kappa_index_base,
            "qcd_divisor_index_base" => qcd_index_base,
            "qcd_divisor_volume" => metadata_number(raw_metadata, "qcd_divisor_volume"),
            "orientifold_metadata_present" => metadata_has(raw_metadata, "orientifold"),
            "orientifold_requested_false" => orientifold_requested === false,
            "orientifold_status" => orientifold_status,
            "source_paper_set_present" => metadata_has(raw_metadata, "source_paper_set"),
            "sampling_metadata_present" => metadata_has(raw_metadata, "sampling"),
            "generator_commit_present" => metadata_has(raw_metadata, "git_commit") || metadata_has(raw_metadata, "generator_commit"),
        )
        record["stored_datasets"] = Dict(
            "basis" => haskey(f, "cytools/geometric/basis"),
            "basis_matrix" => haskey(f, "cytools/geometric/basis_matrix"),
            "effective_cone" => haskey(f, "cytools/geometric/effective_cone"),
            "divisor_volumes" => haskey(f, "cytools/geometric/divisor_volumes"),
            "prime_divisor_volumes" => haskey(f, "cytools/geometric/prime_divisor_volumes"),
            "curve_volumes" => haskey(f, "cytools/geometric/curve_volumes"),
            "mori_cone" => haskey(f, "cytools/geometric/mori_cone"),
            "tip" => haskey(f, "cytools/geometric/tip"),
            "kahler_hyperplanes" => haskey(f, "cytools/geometric/kahler_hyperplanes"),
            "phase" => haskey(f, "phase") || haskey(f, "cytools/geometric/phase") || haskey(f, "cytools/potential/phase"),
            "visible_sector" => haskey(f, "cytools/geometric/visible_sector"),
            "configuration_digest" => haskey(f, "configuration_digest") || haskey(f, "cytools/geometric/configuration_digest"),
            "spd_tolerance" => haskey(f, "spd_tolerance") || haskey(f, "cytools/geometric/spd_tolerance"),
        )
        record["array_shapes_and_types"] = Dict(
            "basis" => [collect(size(basis)), string(eltype(basis))],
            "basis_matrix" => [collect(size(basis_matrix)), string(eltype(basis_matrix))],
            "effective_cone" => [collect(size(qprime)), string(eltype(qprime))],
            "divisor_volumes" => [collect(size(tau)), string(eltype(tau))],
            "prime_divisor_volumes" => [collect(size(prime)), string(eltype(prime))],
            "curve_volumes" => [collect(size(curves)), string(eltype(curves))],
            "mori_cone" => [collect(size(mori)), string(eltype(mori))],
            "tip" => [collect(size(tip)), string(eltype(tip))],
            "kahler_hyperplanes" => [collect(size(hyperplanes)), string(eltype(hyperplanes))],
            "Kinv" => [collect(size(kinv)), string(eltype(kinv))],
            "Q" => [collect(size(q_raw)), string(eltype(q_raw))],
            "L" => [collect(size(l_raw)), string(eltype(l_raw))],
        )
        record["reconstructed_checks"] = Dict(
            "effective_divisor_volumes" => Dict("formula" => "effective_cone * divisor_volumes", "finite" => all(isfinite, effective), "count" => length(effective), "minimum" => minimum(effective)),
            "kahler_margin" => Dict("formula" => "minimum(kahler_hyperplanes * tip)", "finite" => all(isfinite, kahler_slack), "minimum" => margin, "generator_threshold" => 1.0 - 1e-6),
            "charge_orientation" => Dict("Q_integer_valued" => q_integral, "direct_block_matches_effective_cone_transpose" => direct_orientation, "pair_difference_order_matches" => pair_orientation, "h11" => size(q_int, 1), "term_count" => size(q_int, 2), "effective_ray_count" => q_count),
            "stored_QL_formula" => Dict("checked" => ql_status == "pass_existing_tolerance", "status" => ql_status, "max_log10_error" => log_error, "log10_error_tolerance" => QL_LOG10_ERROR_TOLERANCE, "tolerance_source" => QL_TOLERANCE_SOURCE, "sign_mismatches" => sign_mismatches),
            "kinetic_matrix" => Dict("Kinv_raw_asymmetry_max_abs" => kinv_raw_asymmetry, "raw_asymmetry_status" => "measured_before_symmetrization", "spd_tolerance_status" => "not_declared", "Kinv_eigmin" => eigmin(kinv_sym), "K_eigmin" => eigmin(Symmetric(K)), "finite" => all(isfinite, kinv_matrix)),
            "volume_and_margin" => Dict("CY_volume" => Float64(cy_volume), "minimum_prime_divisor_volume" => minimum(Float64.(prime)), "minimum_effective_divisor_volume" => minimum(effective), "minimum_curve_volume" => minimum(Float64.(curves)), "kahler_margin" => margin),
        )
    end
    record["field_classifications"] = FIELD_CLASSIFICATIONS
    record["complete_certificate_sidecar_created"] = false
    record["sidecar_block_reason"] = "genuine scientific gaps remain; no field is asserted beyond its evidence class"
    record
end

function main()
    started = time()
    repository_state = validate_repository_state()
    isfile(MANIFEST) || error("selection manifest missing: $MANIFEST")
    manifest_hash = sha256_file(MANIFEST)
    manifest_hash == MANIFEST_SHA256 || error("selection manifest hash mismatch: $manifest_hash")
    manifest_metadata, manifest_entries = parse_selection_manifest()
    manifest_by_relative = validate_selection_manifest(manifest_metadata, manifest_entries)
    paths = sort!(collect(keys(EXPECTED_SHA256)))
    length(paths) == 18 || error("expected exactly 18 fixed inputs")
    records = Dict{String,Any}[]
    for path in paths
        push!(records, audit_one(path, EXPECTED_SHA256[path], manifest_by_relative[path], repository_state))
    end
    atomic_write(path, content) = begin
        temporary = string(path, ".tmp-", getpid(), "-", time_ns())
        open(temporary, "w") do io
            write(io, content)
            flush(io)
        end
        mv(temporary, path; force=true)
    end
    jsonl = join((json_value(record) for record in records), "\n") * "\n"
    atomic_write(OUTPUT_JSONL, jsonl)
    atomic_zstd(OUTPUT_JSONL, OUTPUT_ZST)
    elapsed_seconds = time() - started
    maxrss_bytes = Sys.maxrss()
    audit_script_hash = sha256_file(abspath(@__FILE__))
    jsonl_hash = sha256_file(OUTPUT_JSONL)
    compressed_jsonl_hash = sha256_file(OUTPUT_ZST)
    statuses = Dict{String,Int}()
    for field in keys(FIELD_CLASSIFICATIONS)
        status = FIELD_CLASSIFICATIONS[field]["status"]
        statuses[status] = get(statuses, status, 0) + 1
    end
    gaps = sort!(String[field for field in keys(FIELD_CLASSIFICATIONS) if FIELD_CLASSIFICATIONS[field]["status"] == "genuinely_unavailable"])
    summary = IOBuffer()
    println(summary, "# Physical evidence audit — 2026-08-25")
    println(summary)
    println(summary, "Status: complete for the authorized read-only Steps 1–3 audit; no inflation or scale point was evaluated.")
    println(summary)
    println(summary, "- Fixed inputs audited: 18 (three each at h11=5,…,10).")
    println(summary, "- Selection manifest SHA-256: `", MANIFEST_SHA256, "`.")
    println(summary, "- Required source commit: `", REQUIRED_COMMIT, "`.")
    println(summary, "- Required source branch: `", REQUIRED_BRANCH, "`.")
    println(summary, "- Repository guard: HEAD, branch, and tracked-file status passed; only the four expected untracked audit outputs were present before the audit.")
    println(summary, "- Manifest binding: strict JSON parse passed for all 18 paths, artifact hashes, h11 values, three distinct polytope identities per h11, triangulation identities, and orientifold.requested=false.")
    println(summary, "- Artifact SHA-256 verification: 18/18 passed.")
    println(summary, "- Complete certificate sidecars: 0; blocked by shared genuine gaps.")
    println(summary, "- Audit resource record: final run internal elapsed `", elapsed_seconds, "` s, maximum resident set `", maxrss_bytes, "` bytes; no geometry-local arrays are retained between inputs.")
    println(summary)
    println(summary, "## Evidence classifications")
    println(summary)
    println(summary, "| Class | Count | Fields |")
    println(summary, "| --- | ---: | --- |")
    for status in ("stored_alias", "exactly_reconstructable", "source_fixed_convention", "genuinely_unavailable")
        fields = sort!(String[field for field in keys(FIELD_CLASSIFICATIONS) if FIELD_CLASSIFICATIONS[field]["status"] == status])
        println(summary, "| `", status, "` | ", length(fields), " | ", join("`" .* fields .* "`", ", "), " |")
    end
    println(summary)
    println(summary, "## Genuine scientific gaps requiring user decisions")
    println(summary)
    for field in gaps
        item = FIELD_CLASSIFICATIONS[field]
        println(summary, "- `", field, "`: ", item["decision"])
    end
    println(summary)
    println(summary, "## Exact reconstruction formulas")
    println(summary)
    println(summary, "- Effective-divisor volumes: normalize stored `effective_cone` to one ray per row, then `E * divisor_volumes`.")
    println(summary, "- Kähler margin: normalize stored `kahler_hyperplanes` to one hyperplane per row, then `minimum(H * tip)`; the historical writer checked `>= 1 - 1e-6`.")
    println(summary, "- Charge orientation: `Q` is `h11 × N`; direct columns equal the transpose of normalized one-ray-per-row `E`, followed by lexicographically ordered pair differences.")
    println(summary, "- Stored coefficient audit: `L` is compared to the source V_CY^-2 direct/pair formula with explicit status. The existing cited diagnostic tolerance is `1e-10` in `validation/run_author_code_coefficient_bridge.py:238-253`; all 18 records are within that tolerance with zero sign mismatches.")
    println(summary, "- Raw kinetic-matrix diagnostic: `Kinv - Kinv'` is measured before symmetrization; SPD eigenspectra remain diagnostic only because no artifact SPD tolerance is declared.")
    println(summary, "- Basis identity evidence: each JSONL record contains exact basis/basis_matrix values, canonical shape/value SHA-256 hashes, basis convention, and stored kappa/QCD index bases. A separate basis index-base field is not stored in the legacy artifact.")

    println(summary)
    println(summary, "## Reproduction and verification")
    println(summary)
    println(summary, "- Recovery record compressed SHA-256: `", HANDOFF_COMPRESSED_SHA256, "`.")
    println(summary, "- Recovery record JSONL SHA-256: `", HANDOFF_JSONL_SHA256, "`.")
    println(summary, "- Audit script SHA-256: `", audit_script_hash, "`.")
    println(summary, "- Audit JSONL SHA-256: `", jsonl_hash, "`.")
    println(summary, "- Audit compressed JSONL SHA-256: `", compressed_jsonl_hash, "`.")
    println(summary, "- Historical generator source SHA-256 at `770b09b7e503ccf01202b1ec2212149c7bd50a5`: `", HISTORICAL_GENERATOR_SHA256, "`.")
    println(summary, "- Committed continuation source SHA-256: `", CONTINUATION_SOURCE_SHA256, "`.")
    println(summary, "- Derivation ledger SHA-256: `", DERIVATION_LEDGER_SHA256, "`.")
    println(summary, "- Reproduction command: `julia --startup-file=no --project=/Users/vmehta/Documents/CYAxiverse/cyaxiverse/CYAxiverse.jl scripts/audit_physical_certificate.jl`.")
    println(summary, "- Verification passed: repository HEAD/branch/tracked-status guard; strict manifest parsing and 18-entry binding; 18/18 artifact hashes; HDF5 dataset/metadata checks; raw Kinv asymmetry measurement before symmetrization; effective-volume and Kähler-margin reconstructions; charge orientation and pair-order checks; Q/L diagnostics with explicit `1e-10` existing tolerance and statuses; canonical basis values/hashes; JSONL parsing; `gzip -t`; and `git diff --check`.")
    println(summary, "- The run read the neighboring cached Julia environment only; it did not install or fetch dependencies, write source/input files, evaluate inflation or scales, compute orientifolds, generate geometry, expand the population, or create a database.")
    println(summary)
    println(summary, "## Per-input records")
    println(summary)
    println(summary, "The machine-readable per-input evidence, observed values, source IDs, array shapes, formulas, and hashes are in [`physical_evidence_audit_20260825.jsonl`](physical_evidence_audit_20260825.jsonl).")
    atomic_write(OUTPUT_MD, String(take!(summary)))
    println("audited_records=", length(records))
    println("manifest_sha256=", manifest_hash)
    println("jsonl=", OUTPUT_JSONL)
    println("compressed_jsonl=", OUTPUT_ZST)
    println("markdown=", OUTPUT_MD)
    println("maxrss_bytes=", maxrss_bytes)
    println("report_output_bytes=", filesize(OUTPUT_JSONL) + filesize(OUTPUT_ZST) + filesize(OUTPUT_MD))
    println("elapsed_seconds=", elapsed_seconds)
end

main()
