#!/usr/bin/env julia

"""Run the bounded, physical-scale diagnostic one geometry at a time.

The runner is deliberately fixed-input.  It validates the approved selection
manifest, sidecar set, source/diff identity, and Julia environment before it
loads a potential.  A completed geometry is represented by one atomic JSON
checkpoint and a sibling checksum.  Resume accepts only a checkpoint whose
identity and checksum reproduce exactly; it never silently replaces an input
or skips an unverified record.

This is a diagnostic pilot.  The evaluator is called with
`max_negative_modes=1`, so branch coverage is explicitly `partial_index_range`.
The control gate remains `not_established`, and no viability, production, or
validated-candidate claim is emitted.
"""

using HDF5
using LinearAlgebra
using SHA
using Printf

const RUNNER_SCHEMA = "physical-scale-inflation-pilot-run-v1"
const CHECKPOINT_SCHEMA = "physical-scale-inflation-geometry-checkpoint-v6"
const ACCOUNTING_SCHEMA = "physical-scale-inflation-terminal-accounting-v1"
const ENVIRONMENT_SCHEMA = "physical-scale-inflation-environment-v1"
const QUARANTINE_SCHEMA = "physical-scale-inflation-nonterminal-partial-quarantine-v1"
const QUARANTINE_VERSION = "nonterminal-partials-v1"
const QUARANTINE_DIR_NAME = "quarantine"
const QUARANTINE_MANIFEST_NAME = "nonterminal_partial_quarantine_manifest-v1.json"
const OLD_CONVERSION_POLICY_VERSION = "kinv-absolute-tolerance-v1"
const OLD_KINV_CONVERSION_RULE = "max_absolute_error <= 1e-12"
const OLD_QUARANTINE_FAILURE_REASON =
    "uncheckpointed header-only partial from old Kinv absolute-only policy"
const CURRENT_QUARANTINE_FAILURE_REASON =
    "uncheckpointed header-only partial from lazy-transpose array-hash failure"

const WORKTREE = "/Users/vmehta/Documents/CYAxiverse/cyaxiverse/CYAxiverse.jl.worktrees/physical-scale-inflation-20260825"
const DATA_ROOT = "/Users/vmehta/Documents/CYAxiverse/cyaxiverse/data"
const PROJECT_ROOT = "/Users/vmehta/Documents/CYAxiverse/cyaxiverse/CYAxiverse.jl"
const SELECTION_MANIFEST = "/private/tmp/cyax-inflation-physical-scale-pilot-20260825/selection_manifest.json"
const SELECTION_SHA256 = "a6df5dca258c11724d4162477cdee7cc34e5802f2f3f296a7ea64b55f23c3247"
const REQUIRED_COMMIT = "9f31d716eaab8d63d3f76826a40de5ae38c7015d"
const REQUIRED_BRANCH = "agents/physical-scale-inflation-20260825"
const PROJECT_SHA256 = "38275eaf04c8f9ae28541542326f6c791c3a6a40d2a9ca3916088ca0652c368c"
const MANIFEST_SHA256 = "80648aa2f1ff5d9d8b6296429ad8e0d53eb5deb2a0e8784668e7dab4954713a5"
const CERTIFICATE_DIR = joinpath(WORKTREE, "validation/physical_scaling_certificates_20260825")
const POLICY_PATH = joinpath(CERTIFICATE_DIR, "physical_scaling_pilot_policy-v1.json")
const POLICY_CHECKSUM_PATH = joinpath(CERTIFICATE_DIR, "physical_scaling_pilot_policy-v1.sha256")
const SIDECAR_CHECKSUM_PATH = joinpath(CERTIFICATE_DIR, "physical_scaling_certificates.sha256")
const VALIDATOR = joinpath(WORKTREE, "scripts/validate_physical_scaling_sidecars_20260825.py")
const CONTINUATION = joinpath(WORKTREE, "scripts/inflation_scale_continuation.jl")
const CHECKPOINT_VALIDATOR = joinpath(WORKTREE, "scripts/validate_physical_scale_checkpoint_20260825.py")
const EXECUTION_SOURCE_MANIFEST_SCHEMA =
    "physical-scale-inflation-execution-source-manifest-v2"
const EXECUTION_SOURCE_MANIFEST_NAME = "execution_source_manifest-v5.json"
const EXECUTION_SOURCE_FILES = (
    "scripts/run_physical_scale_inflation_pilot_20260825.jl",
    "scripts/validate_physical_scale_checkpoint_20260825.py",
    "scripts/validate_physical_scaling_sidecars_20260825.py",
    "scripts/generate_physical_scaling_sidecars_20260825.jl",
    "scripts/preflight_physical_scaling_20260825.jl",
    "scripts/inflation_scale_continuation.jl",
    "scripts/inflation_scan_shards_common.jl",
    "scripts/inflation_diagnostics_common.jl",
    "scripts/inflation_refinement_common.jl",
    "scripts/audit_physical_certificate.jl",
    "src/CYAxiverse.jl",
    "src/read.jl",
    "src/generate.jl",
    "src/structs.jl",
    "Project.toml",
    "../../CYAxiverse.jl/Manifest.toml",
)
const OUTPUT_ROOT_DEFAULT = joinpath(WORKTREE, "validation/physical_scale_inflation_pilot_20260825")
const CHECKPOINT_DIR_NAME = "checkpoints"
const SUMMARY_FILE_NAME = "summary.csv"
const SHARD_DIR_NAME = "shards"
const SCALE_STRINGS = ("0.9", "0.95", "0.99", "1.0", "1.01", "1.05", "1.1")
const SCALE_VALUES = Float64.(parse.(Float64, SCALE_STRINGS))
const MAX_RSS_BYTES = Int64(2_000_000_000)
const MAX_NEW_OUTPUT_BYTES = Int64(2_000_000_000)
const MAX_BRANCHES = 400_000
const MAX_STAGE_ALLOCATED_BYTES = 750_000_000
const MAX_NEGATIVE_MODES = 1
const CORRECTION_TOLERANCE = 1e-9
const CORRECTION_ITERATIONS = 100
const MATCHING_TOLERANCE = 0.1
const DUPLICATE_TOLERANCE = 1e-7
const ZERO_EIGENVALUE_TOLERANCE = 1e-8
const PHASE = "zero_phases"
const UNITS = "M_s=M_Pl;k=dimensionless"
const NORMALIZATION = "homogeneous_full_volume_k32"
const SPD_TOLERANCE = "1e-12"
const PRECISION_BITS = 256
const COVERAGE_LABEL = "partial_index_range"
const DOMAIN_CERTIFICATE_VERSION = "physical-domain-certificate-3"
const CONVERSION_POLICY_VERSION = "kinv-mixed-tolerance-v1"
const KINV_CONVERSION_RULE =
    "max_absolute_error <= 1e-12 OR max_relative_error <= 1e-12"

# Load the current evaluator exactly as implemented.  The runner does not
# call its discovery/CSV driver; it calls the one-geometry evaluator directly.
include(CONTINUATION)

const SELECTION_PARSER = raw"""
import json, re, sys
path, expected_sha, expected_commit, expected_branch = sys.argv[1:]
with open(path, encoding='utf-8') as stream:
    data = json.load(stream)
if data.get('manifest_schema') != 'cyax-inflation-physical-scale-pilot-selection-1': raise SystemExit('selection schema mismatch')
if data.get('selected_count') != 18 or not isinstance(data.get('selected_inputs'), list) or len(data['selected_inputs']) != 18: raise SystemExit('selection count mismatch')
seen = set()
for i, item in enumerate(data['selected_inputs']):
    required = ('path','sha256','h11','polytope','polytope_id','triangulation_id','orientifold')
    if any(k not in item for k in required): raise SystemExit(f'entry {i} missing identity')
    if item['path'] in seen: raise SystemExit('duplicate selection path')
    seen.add(item['path'])
    if not re.fullmatch(r'[0-9a-f]{64}', item['sha256']): raise SystemExit('invalid input hash')
    if item['orientifold'].get('requested') is not False: raise SystemExit('orientifold input is not false')
    print('ENTRY\t' + '\t'.join((item['path'], item['sha256'], str(item['h11']), str(item['polytope']), '1', item['polytope_id'], item['triangulation_id'])))
print('COUNT\t18')
"""

const SIDECAR_PARSER = raw"""
import json, sys
path = sys.argv[1]
o = json.load(open(path, encoding='utf-8'))
i = o['input']; e = o['geometry_evidence']; c = o['approved_conventions']
s = o['source_identity']; d = o['configuration_digest']
hashes = e['array_hashes']
fields = [
    path, i['relative_path'], i['artifact_sha256'], str(i['h11']), str(i['polytope']), str(i['frst']),
    i['polytope_id'], i['triangulation_id'], str(i['orientifold_requested']).lower(),
    c['phase_convention'], c['units'], c['normalization'], str(c['certificate_precision_bits']), c['spd_tolerance'],
    'reconstructed-replay-identity-v1:sha256:' + s['digest_sha256'], d['geometry_sha256'], d['common_sha256'],
    s['current_continuation_source_sha256'], s['current_complete_git_diff_sha256'],
    o['physical_scaling_gate']['status'], o['physical_control_gate']['status'], o['physical_viability']['status'],
    s['geometry_sha256'], s['selection_manifest_sha256'], s['polytope_id'], s['triangulation_id'],
    hashes['tau'], hashes['Kinv_raw'], hashes['CY_volume'], hashes['prime_divisor_volumes'], hashes['curve_volumes'],
    hashes['tip'], hashes['effective_cone_normalized'], hashes['kahler_hyperplanes_normalized'], hashes['Q'], hashes['L'], hashes['K'],
    hashes['effective_divisor_volumes'], hashes['kahler_slack'], str(o['stored_QL_reference']['max_log10_error']), str(o['stored_QL_reference']['sign_mismatches'])]
print('SIDECAR\t' + '\t'.join(fields))
"""

const CHECKPOINT_PARSER = raw"""
import hashlib, json, pathlib, sys
path, expected_schema, expected_policy, expected_selection, expected_common, expected_diff = sys.argv[1:]
p = pathlib.Path(path); checksum = p.with_suffix(p.suffix + '.sha256')
if not p.is_file() or not checksum.is_file(): raise SystemExit('checkpoint or checksum missing')
fields = checksum.read_text(encoding='utf-8').split()
if len(fields) != 2 or fields[1] != p.name or fields[0] != hashlib.sha256(p.read_bytes()).hexdigest(): raise SystemExit('checkpoint checksum mismatch')
o = json.loads(p.read_text(encoding='utf-8'))
if o.get('schema') != expected_schema or o.get('policy_sha256') != expected_policy or o.get('selection_manifest_sha256') != expected_selection or o.get('common_configuration_digest_sha256') != expected_common or o.get('current_complete_git_diff_sha256') != expected_diff: raise SystemExit('checkpoint policy identity mismatch')
if o.get('terminal_status') != 'completed' or o.get('terminal_geometry_count') != 1 or o.get('terminal_scale_point_count') != 7: raise SystemExit('checkpoint terminal count mismatch')
if o.get('scale_strings') != ['0.9','0.95','0.99','1.0','1.01','1.05','1.1']: raise SystemExit('checkpoint scale strings mismatch')
if o.get('physical_control_gate', {}).get('status') != 'not_established': raise SystemExit('checkpoint control status mismatch')
if o.get('physical_viability', {}).get('claim') is not False or o.get('production_qualified') is not False or o.get('validated') is not False: raise SystemExit('checkpoint contains an unauthorized claim')
rows = o.get('scale_records')
if not isinstance(rows, list) or len(rows) != 7: raise SystemExit('checkpoint scale record count mismatch')
for row, scale in zip(rows, ['0.9','0.95','0.99','1.0','1.01','1.05','1.1']):
    if row.get('scale_string') != scale or row.get('physical_scaling_gate_status') != 'passed' or row.get('physical_control_gate_status') != 'not_established': raise SystemExit('checkpoint scale identity/status mismatch')
print('CHECKPOINT\t' + hashlib.sha256(p.read_bytes()).hexdigest())
"""

const MANIFEST_PARSER = raw"""
import hashlib, json, pathlib, sys
path, checksum_path, expected_schema, expected_policy, expected_selection = sys.argv[1:]
p=pathlib.Path(path); c=pathlib.Path(checksum_path)
fields=c.read_text(encoding='utf-8').split()
if len(fields)!=2 or fields[1]!=p.name or fields[0]!=hashlib.sha256(p.read_bytes()).hexdigest(): raise SystemExit('manifest checksum mismatch')
o=json.loads(p.read_text(encoding='utf-8'))
if o.get('schema')!=expected_schema or o.get('policy_sha256')!=expected_policy or o.get('selection_manifest_sha256')!=expected_selection: raise SystemExit('manifest identity mismatch')
if o.get('scale_strings')!=['0.9','0.95','0.99','1.0','1.01','1.05','1.1']: raise SystemExit('manifest scale strings mismatch')
if o.get('physical_control_gate')!='not_established' or o.get('production_qualified') is not False or o.get('validated') is not False: raise SystemExit('manifest claims unauthorized status')
print('MANIFEST\t' + hashlib.sha256(p.read_bytes()).hexdigest())
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
json_value(::Missing) = "null"
json_value(x::Bool) = x ? "true" : "false"
json_value(x::Integer) = string(x)
json_value(x::AbstractFloat) = isfinite(x) ? repr(x) : string('"', json_escape(string(x)), '"')
json_value(x::Symbol) = string('"', json_escape(string(x)), '"')
json_value(x::AbstractString) = string('"', json_escape(x), '"')
json_value(x::Tuple) = json_value(collect(x))
json_value(x::AbstractArray) = string('[', join(json_value.(collect(x)), ','), ']')
json_value(x::NamedTuple) = json_value(Dict(String(k) => getfield(x, k) for k in keys(x)))
json_value(x::Pair) = json_value(Dict(string(first(x)) => last(x)))
function json_value(x::AbstractDict)
    keys_sorted = sort!(String[string(k) for k in keys(x)])
    string('{', join((string(json_value(k), ':', json_value(x[k])) for k in keys_sorted), ','), '}')
end
json_value(x) = string('"', json_escape(string(x)), '"')

sha256_bytes(bytes::Vector{UInt8}) = bytes2hex(sha256(bytes))
sha256_file(path::AbstractString) = sha256_bytes(read(path))

function complete_git_diff_sha256()
    sha256_bytes(read(`git -C $WORKTREE diff --binary HEAD`))
end

function execution_source_hashes()
    hashes = Dict{String,Any}()
    for relative in EXECUTION_SOURCE_FILES
        path = normpath(joinpath(WORKTREE, relative))
        isfile(path) || error("execution-critical source is missing: $relative")
        hashes[relative] = sha256_file(path)
    end
    hashes
end

function execution_source_manifest_payload()
    source_hashes = execution_source_hashes()
    payload = Dict{String,Any}(
        "schema" => EXECUTION_SOURCE_MANIFEST_SCHEMA,
        "manifest_version" => 2,
        "worktree" => WORKTREE,
        "source_file_inventory" => collect(EXECUTION_SOURCE_FILES),
        "source_file_hashes" => source_hashes,
        "complete_git_diff_sha256" => complete_git_diff_sha256(),
        "project_sha256" => PROJECT_SHA256,
        "manifest_sha256" => MANIFEST_SHA256,
        "method" => "sha256 of each immutable execution source; payload hash excludes payload_sha256; file hash binds the canonical bytes",
    )
    payload["payload_sha256"] = sha256_bytes(Vector{UInt8}(codeunits(json_value(payload))))
    payload
end

function write_execution_source_manifest(root)
    path = joinpath(root, EXECUTION_SOURCE_MANIFEST_NAME)
    payload = execution_source_manifest_payload()
    digest, written = hashed_json_write(path, payload)
    (; path, digest, written, source_hashes=payload["source_file_hashes"],
       complete_diff_sha256=payload["complete_git_diff_sha256"])
end

function validate_live_execution_source_manifest(path, expected_digest=nothing)
    isfile(path) || error("execution-source manifest is missing: $path")
    digest = sha256_file(path)
    expected_digest === nothing || digest == expected_digest ||
        error("execution-source manifest hash changed")
    output = run_python_file(CHECKPOINT_VALIDATOR, "--source-only", path,
        "--source-manifest-sha256", digest)
    fields = split(chomp(output), '\t'; keepempty=true)
    length(fields) == 3 && fields[1] == "SOURCE" || error("execution-source manifest validator failed")
    (; path, digest, source_hashes=execution_source_hashes(),
       complete_diff_sha256=complete_git_diff_sha256())
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
    length(bytes)
end

atomic_write(path::AbstractString, text::AbstractString) = atomic_write(path, Vector{UInt8}(codeunits(text)))

function atomic_copy(source::AbstractString, destination::AbstractString)
    bytes = read(source)
    atomic_write(destination, bytes)
    length(bytes)
end

function output_tree_bytes(root::AbstractString)
    isdir(root) || return Int64(0)
    total = Int64(0)
    for (directory, _, files) in walkdir(root)
        for file in files
            total += Int64(filesize(joinpath(directory, file)))
        end
    end
    total
end

function hashed_json_write(path::AbstractString, payload)
    text = json_value(payload) * "\n"
    bytes = Vector{UInt8}(codeunits(text))
    atomic_write(path, bytes)
    digest = sha256_bytes(bytes)
    checksum = string(digest, "  ", basename(path), "\n")
    atomic_write(string(path, ".sha256"), checksum)
    digest, length(bytes) + sizeof(codeunits(checksum))
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

function runner_git(args...)
    strip(read(`git -C $WORKTREE $args`, String))
end

function run_python(script, args...)
    read(`python3 -c $script $args`, String)
end

function run_python_file(path::AbstractString, args...)
    read(`python3 $path $args`, String)
end

function validate_environment()
    VERSION == v"1.12.6" || error("Julia version mismatch: $(VERSION)")
    active = Base.active_project()
    active === nothing && error("active Julia project is missing")
    realpath(active) == joinpath(PROJECT_ROOT, "Project.toml") ||
        error("active Julia project is not the pinned environment: $active")
    project = joinpath(PROJECT_ROOT, "Project.toml")
    manifest = joinpath(PROJECT_ROOT, "Manifest.toml")
    sha256_file(project) == PROJECT_SHA256 || error("Project.toml hash mismatch")
    sha256_file(manifest) == MANIFEST_SHA256 || error("Manifest.toml hash mismatch")
    sha256_file(joinpath(WORKTREE, "Project.toml")) == PROJECT_SHA256 ||
        error("worktree Project.toml hash mismatch")
    strip(get(ENV, "CYAXIVERSE_DATA_DIR", "")) == DATA_ROOT ||
        error("CYAXIVERSE_DATA_DIR must be the fixed data root")
    isempty(strip(get(ENV, "newARGS", ""))) || error("newARGS would alter data routing")
    isempty(strip(get(ENV, "PYTHON", ""))) || error("PYTHON is not allowed for this evaluator")
    (; julia_version=string(VERSION), julia_executable=abspath(joinpath(Sys.BINDIR, Base.julia_exename())),
       active_project=realpath(active), project_sha256=sha256_file(project),
       manifest_sha256=sha256_file(manifest), worktree_project_sha256=sha256_file(joinpath(WORKTREE, "Project.toml")),
       data_root=DATA_ROOT, environment_variables=Dict(k => get(ENV, k, "") for k in
           ("CYAXIVERSE_DATA_DIR", "JULIA_PROJECT", "JULIA_DEPOT_PATH", "PYTHON", "newARGS")))
end

function fixed_inputs()
    sha256_file(SELECTION_MANIFEST) == SELECTION_SHA256 || error("selection manifest hash mismatch")
    output = run_python(SELECTION_PARSER, SELECTION_MANIFEST, SELECTION_SHA256, REQUIRED_COMMIT, REQUIRED_BRANCH)
    lines = split(chomp(output), '\n'; keepempty=false)
    count(line -> startswith(line, "ENTRY\t"), lines) == 18 || error("selection parser did not return 18 inputs")
    entries = NamedTuple[]
    for line in filter(line -> startswith(line, "ENTRY\t"), lines)
        fields = split(line, '\t'; keepempty=true)
        length(fields) == 8 || error("malformed fixed selection entry")
        push!(entries, (path=fields[2], artifact_sha256=fields[3], h11=parse(Int, fields[4]),
            polytope=parse(Int, fields[5]), frst=parse(Int, fields[6]), polytope_id=fields[7],
            triangulation_id=fields[8], orientifold_requested=false))
    end
    all(entry -> startswith(entry.path, DATA_ROOT * "/") && entry.frst == 1,
        entries) || error("selection includes a path outside the fixed data root or non-frst-1 input")
    all(entry -> entry.h11 in 5:10 && entry.polytope in 1:3, entries) ||
        error("selection h11/polytope identity is outside the fixed 18")
    all(h11 -> count(entry -> entry.h11 == h11, entries) == 3, 5:10) ||
        error("selection does not contain three inputs at each h11")
    entries
end

function validate_sidecars()
    isfile(VALIDATOR) || error("sidecar validator is missing")
    result = run_python_file(VALIDATOR, CERTIFICATE_DIR, POLICY_PATH, SELECTION_MANIFEST,
        DATA_ROOT, WORKTREE)
    lines = split(chomp(result), '\n'; keepempty=false)
    length(lines) == 19 && startswith(lines[1], "META\t") ||
        error("sidecar validator did not return 18 rows")
    meta = split(lines[1], '\t'; keepempty=true)
    length(meta) == 3 || error("malformed sidecar validator metadata")
    rows = NamedTuple[]
    for line in lines[2:end]
        fields = split(line, '\t'; keepempty=true)
        length(fields) == 15 && fields[1] == "ENTRY" || error("malformed sidecar validator row")
        push!(rows, (name=fields[2], relative_path=fields[3], artifact_sha256=fields[4],
            h11=parse(Int, fields[5]), polytope=parse(Int, fields[6]), polytope_id=fields[7],
            triangulation_id=fields[8], sidecar_sha256=fields[9], scaling_status=fields[10],
            control_status=fields[11], ql_error=parse(Float64, fields[12]),
            ql_sign_mismatches=parse(Int, fields[13]), basis_sha256=fields[14],
            basis_matrix_sha256=fields[15]))
    end
    length(rows) == 18 || error("expected 18 validated sidecars")
    policy_sha = sha256_file(POLICY_PATH)
    policy_checksum = split(strip(read(POLICY_CHECKSUM_PATH, String)))
    length(policy_checksum) == 2 && policy_checksum[1] == policy_sha &&
        policy_checksum[2] == basename(POLICY_PATH) || error("policy checksum mismatch")
    complete_git_diff_sha256() == meta[3] || error("complete Git diff hash changed since sidecar generation")
    (; common_digest=meta[2], complete_diff_sha256=meta[3], policy_sha256=policy_sha, rows)
end

function sidecar_meta(path::AbstractString)
    out = run_python(SIDECAR_PARSER, path)
    fields = split(chomp(out), '\t'; keepempty=true)
    length(fields) == 42 && fields[1] == "SIDECAR" || error("malformed sidecar metadata: $path")
    (path=fields[2], relative_path=fields[3], artifact_sha256=fields[4], h11=parse(Int, fields[5]),
        polytope=parse(Int, fields[6]), frst=parse(Int, fields[7]), polytope_id=fields[8],
        triangulation_id=fields[9], orientifold_requested=fields[10] == "false", phase=fields[11],
        units=fields[12], normalization=fields[13], precision_bits=parse(Int, fields[14]),
        spd_tolerance=fields[15], source_identity=fields[16], geometry_digest=fields[17],
        common_digest=fields[18], continuation_sha256=fields[19], diff_sha256=fields[20],
        scaling_status=fields[21], control_status=fields[22], viability_status=fields[23],
        source_geometry_sha256=fields[24], source_selection_sha256=fields[25],
        source_polytope_id=fields[26], source_triangulation_id=fields[27],
        hashes=Dict(k => fields[i] for (i, k) in zip(28:40, ("tau", "Kinv_raw", "CY_volume",
            "prime_divisor_volumes", "curve_volumes", "tip", "effective_cone_normalized",
            "kahler_hyperplanes_normalized", "Q", "L", "K", "effective_divisor_volumes",
            "kahler_slack"))), ql_error=parse(Float64, fields[41]), ql_sign_mismatches=parse(Int, fields[42]))
end

function geometry_sidecar(entry)
    name = string("h11_", lpad(entry.h11, 3, '0'), "_np_", lpad(entry.polytope, 7, '0'),
        "_cy_", lpad(entry.frst, 7, '0'), ".physical-scaling-certificate-v1.json")
    path = joinpath(CERTIFICATE_DIR, name)
    meta = sidecar_meta(path)
    meta.relative_path == relpath(entry.path, DATA_ROOT) || error("sidecar/input path mismatch")
    meta.artifact_sha256 == entry.artifact_sha256 || error("sidecar/input artifact hash mismatch")
    meta.h11 == entry.h11 && meta.polytope == entry.polytope && meta.frst == entry.frst ||
        error("sidecar/input geometry identity mismatch")
    meta.scaling_status == "passed" || error("physical scaling sidecar gate is not passed")
    meta.control_status == "not_established" || error("control gate is not the approved independent status")
    meta.viability_status == "not_evaluated" || error("sidecar contains a viability result")
    sha256_file(entry.path) == entry.artifact_sha256 || error("fixed geometry artifact changed")
    sha256_file(path) == only(filter(row -> row.name == name,
        validate_sidecars().rows)).sidecar_sha256 || error("sidecar checksum binding mismatch")
    meta
end

function load_geometry_evidence(entry, meta)
    geom = GeometryIndex(entry.h11, entry.polytope, entry.frst)
    loaded = CYAxiverse.read.oriented_potential(geom)
    path = entry.path
    evidence = h5open(path, "r") do file
        read_data(name) = haskey(file, name) ? read(file[name]) : error("missing fixed dataset $name")
        tau = Float64.(read_data("cytools/geometric/divisor_volumes"))
        prime = Float64.(read_data("cytools/geometric/prime_divisor_volumes"))
        curves = Float64.(read_data("cytools/geometric/curve_volumes"))
        tip = Float64.(read_data("cytools/geometric/tip"))
        qprime = Float64.(read_data("cytools/geometric/effective_cone"))
        hyperplanes = Float64.(read_data("cytools/geometric/kahler_hyperplanes"))
        kinv = Float64.(read_data("cytools/geometric/Kinv"))
        volume = Float64(read_data("cytools/geometric/CY_volume"))
        Q_raw = read_data("cytools/potential/Q")
        L_raw = Float64.(read_data("cytools/potential/L"))
        Q = Int.(round.(Q_raw)); all(Q_raw .== Q) || error("noninteger Q in fixed artifact")
        # Materialize both normalized orientations.  The sidecar generator
        # hashes a concrete Matrix{Float64}; leaving the transposed branch as
        # a lazy Adjoint can select a different BLAS accumulation path and
        # produce a different last-bit canonical hash.
        E = size(qprime, 2) == entry.h11 ? Matrix{Float64}(qprime) :
            Matrix{Float64}(qprime')
        H = size(hyperplanes, 2) == length(tip) ? Matrix{Float64}(hyperplanes) :
            Matrix{Float64}(hyperplanes')
        effective = E * tau
        kahler_slack = H * tip
        (; tau, prime, curves, tip, E, H, kinv, volume, Q, L=L_raw,
            effective, kahler_slack, margin=minimum(kahler_slack))
    end
    arrays = Dict(
        "tau" => evidence.tau, "Kinv_raw" => evidence.kinv, "CY_volume" => [evidence.volume],
        "prime_divisor_volumes" => evidence.prime, "curve_volumes" => evidence.curves,
        "tip" => evidence.tip, "effective_cone_normalized" => evidence.E,
        "kahler_hyperplanes_normalized" => evidence.H, "Q" => evidence.Q, "L" => evidence.L,
        "K" => Matrix(loaded.K), "effective_divisor_volumes" => evidence.effective,
        "kahler_slack" => evidence.kahler_slack)
    for (key, value) in arrays
        canonical_array_sha256(value) == meta.hashes[key] ||
            error("sidecar array hash mismatch for $key in $(meta.relative_path)")
    end
    loaded.Q == evidence.Q && loaded.L == evidence.L ||
        error("current evaluator canonicalized the fixed input; identity is ambiguous")
    loaded.K == Matrix(inv(Symmetric(evidence.kinv))) ||
        error("current evaluator K differs from sidecar reconstruction")
    geometry_data = (; τ_volumes=evidence.tau, kinv=evidence.kinv, cy_volume=evidence.volume,
        prime_divisor_volumes=evidence.prime, effective_divisor_volumes=evidence.effective,
        curve_volumes=evidence.curves, kahler_margin=evidence.margin,
        basis_identity="sidecar-basis-sha256:" * only(filter(row -> row.name == basename(meta.path),
            validate_sidecars().rows)).basis_sha256,
        charge_orientation="Q is h11 x N; direct columns are transpose(E), then ordered pair differences",
        phase_convention=meta.phase, units=meta.units, normalization=meta.normalization,
        source_identity=meta.source_identity, configuration_digest=meta.geometry_digest,
        moduli_status=:not_established, instanton_control=:not_established,
        perturbative_control=:not_established, visible_sector_status=:not_applicable,
        spd_tolerance=BigFloat(meta.spd_tolerance), precision_bits=meta.precision_bits)
    (; geom=GeometryIndex(entry.h11, entry.polytope, entry.frst), Q=loaded.Q,
       L=loaded.L, K=Matrix(loaded.K), geometry_data, evidence)
end

function runner_context_to_dict(value)
    value isa NamedTuple && return Dict(String(k) => runner_context_to_dict(getfield(value, k)) for k in keys(value))
    value isa AbstractDict && return Dict(String(k) => runner_context_to_dict(v) for (k, v) in value)
    value isa Tuple && return [runner_context_to_dict(v) for v in value]
    value isa AbstractArray && return [runner_context_to_dict(v) for v in value]
    value isa Symbol && return string(value)
    value
end

function prohibited_actions()
    Dict{String,Any}(
        "orientifold_computed" => false,
        "geometry_generated" => false,
        "population_expanded" => false,
        "replacement_or_silent_skip" => false,
        "database_written" => false,
        "dependency_specification_changed" => false,
        "commit_created" => false,
        "exhaustive_coverage_claim" => false,
        "physical_viability_claim" => false,
        "production_claim" => false,
        "validated_candidate_claim" => false,
    )
end

function canonical_run_configuration(entry, meta, identity;
        execution_source_manifest_sha256="", execution_source_hashes=Dict{String,Any}())
    Dict{String,Any}(
        "configuration_schema" => "physical-scale-inflation-run-configuration-v1",
        "fixed_input" => Dict{String,Any}(
            "relative_path" => relpath(entry.path, DATA_ROOT),
            "artifact_sha256" => entry.artifact_sha256,
            "h11" => entry.h11, "polytope" => entry.polytope, "frst" => entry.frst,
            "polytope_id" => entry.polytope_id, "triangulation_id" => entry.triangulation_id,
            "orientifold_requested" => false,
        ),
        "selection_manifest_sha256" => SELECTION_SHA256,
        "policy_sha256" => identity.policy_sha256,
        "common_configuration_digest_sha256" => identity.common_digest,
        "continuation_source_sha256" => meta.continuation_sha256,
        "current_complete_git_diff_sha256" => identity.complete_diff_sha256,
        "execution_source_manifest_sha256" => execution_source_manifest_sha256,
        "execution_source_hashes" => execution_source_hashes,
        "scale_strings" => collect(SCALE_STRINGS),
        "max_negative_modes" => MAX_NEGATIVE_MODES,
        "phase_convention" => meta.phase,
        "units" => meta.units,
        "normalization" => meta.normalization,
        "precision_bits" => meta.precision_bits,
        "domain_certificate_version" => DOMAIN_CERTIFICATE_VERSION,
        "conversion_policy_version" => CONVERSION_POLICY_VERSION,
        "kinv_conversion_acceptance" => KINV_CONVERSION_RULE,
        "coverage_label" => COVERAGE_LABEL,
        "resource_limits" => Dict{String,Any}(
            "maximum_resident_bytes" => MAX_RSS_BYTES,
            "maximum_new_output_bytes" => MAX_NEW_OUTPUT_BYTES,
            "one_geometry_at_a_time" => true,
            "atomic_checkpoint_after_each_geometry" => true,
            "idempotent_resume" => true,
        ),
        "prohibitions" => prohibited_actions(),
    )
end

run_configuration_digest(config) = sha256_bytes(Vector{UInt8}(codeunits(json_value(config))))

function checkpoint_name(entry)
    string("h11_", lpad(entry.h11, 3, '0'), "_np_", lpad(entry.polytope, 7, '0'),
        "_cy_", lpad(entry.frst, 7, '0'), ".checkpoint-v6.json")
end

checkpoint_path(root, entry) = joinpath(root, CHECKPOINT_DIR_NAME, checkpoint_name(entry))

function geometry_output_name(entry)
    string("h11_", lpad(entry.h11, 3, '0'), "_np_", lpad(entry.polytope, 7, '0'),
        "_cy_", lpad(entry.frst, 7, '0'))
end

geometry_identity_value(entry) = entry.h11 * 1_000_000 + entry.polytope * 1_000 + entry.frst

geometry_output_root(root, entry) = joinpath(root, "geometries", geometry_output_name(entry))

geometry_shard_name(entry) = string("inflation_scale_continuation_shard_", lpad(entry.h11, 3, '0'),
    "_np_", lpad(entry.polytope, 7, '0'), "_cy_", lpad(entry.frst, 7, '0'), ".csv")

quarantine_root(root, entry) = joinpath(root, QUARANTINE_DIR_NAME, QUARANTINE_VERSION,
    geometry_output_name(entry))

quarantine_manifest_path(root) = joinpath(root, QUARANTINE_DIR_NAME, QUARANTINE_MANIFEST_NAME)

function old_policy_header(fields)
    join(string.(filter(field -> field != :conversion_policy_version, fields)), ",") * "\n"
end

# The current runner writes the mixed-tolerance policy column.  Quarantine is
# allowed to adopt a header-only partial produced by either the superseded
# absolute-only runner or the current runner, but the manifest binds which one
# was actually observed.
current_policy_header(fields) = join(string.(fields), ",") * "\n"

function validate_old_header_only_csv(path::AbstractString, fields, role::AbstractString)
    isfile(path) || error("quarantine source file is missing: $path")
    islink(path) && error("quarantine source file must not be a symlink: $path")
    bytes = read(path)
    text = try
        String(copy(bytes))
    catch
        error("quarantine source file is not UTF-8: $path")
    end
    partial_policy_version, partial_kinv_rule, failure_reason =
        if text == old_policy_header(fields)
            (OLD_CONVERSION_POLICY_VERSION, OLD_KINV_CONVERSION_RULE,
                OLD_QUARANTINE_FAILURE_REASON)
        elseif text == current_policy_header(fields)
            (CONVERSION_POLICY_VERSION, KINV_CONVERSION_RULE,
                CURRENT_QUARANTINE_FAILURE_REASON)
        else
            error("quarantine source $role is not a recognized header-only CSV: $path")
        end
    (; path=String(path), sha256=sha256_bytes(bytes), bytes=Int64(length(bytes)), data_rows=0,
       role=role, header_sha256=sha256_bytes(bytes),
       partial_policy_version=partial_policy_version, partial_kinv_rule=partial_kinv_rule,
       failure_reason=failure_reason)
end

function output_files_at(target)
    isdir(target) || return String[]
    files = String[]
    for (directory, _, names) in walkdir(target)
        for name in sort!(names)
            path = joinpath(directory, name)
            isfile(path) || error("unexpected non-file in uncheckpointed output: $path")
            islink(path) && error("unexpected symlink in uncheckpointed output: $path")
            push!(files, path)
        end
    end
    sort!(files)
end

partial_output_files(root, entry) = output_files_at(geometry_output_root(root, entry))

function quarantine_manifest_fields(root, entry, identity, source_manifest, entries, completed, files)
    isempty(files) && error("quarantine file inventory is empty")
    partial_policy_version = files[1].partial_policy_version
    partial_kinv_rule = files[1].partial_kinv_rule
    partial_failure_reason = files[1].failure_reason
    all(file -> file.partial_policy_version == partial_policy_version &&
        file.partial_kinv_rule == partial_kinv_rule &&
        file.failure_reason == partial_failure_reason, files) ||
        error("quarantine files do not share one partial-policy identity")
    Dict{String,Any}(
        "schema" => QUARANTINE_SCHEMA,
        "manifest_version" => 1,
        "quarantine_version" => QUARANTINE_VERSION,
        "status" => "quarantined",
        "run_id" => "physical-scale-inflation-pilot-20260825",
        "geometry_index" => geometry_identity_value(entry),
        "geometry" => Dict{String,Any}("h11" => entry.h11, "polytope" => entry.polytope,
            "frst" => entry.frst, "polytope_id" => entry.polytope_id,
            "triangulation_id" => entry.triangulation_id),
        "geometry_artifact_sha256" => entry.artifact_sha256,
        "selection_manifest_sha256" => SELECTION_SHA256,
        "policy_sha256" => identity.policy_sha256,
        "common_configuration_digest_sha256" => identity.common_digest,
        "current_complete_git_diff_sha256" => identity.complete_diff_sha256,
        "execution_source_manifest_sha256" => source_manifest.digest,
        "execution_source_hashes" => source_manifest.source_hashes,
        "old_conversion_policy_version" => partial_policy_version,
        "old_kinv_conversion_acceptance" => partial_kinv_rule,
        "failure_reason" => partial_failure_reason,
        "coverage_label" => COVERAGE_LABEL,
        "max_negative_modes" => MAX_NEGATIVE_MODES,
        "original_geometry_output_root" => relpath(geometry_output_root(root, entry), root),
        "quarantine_geometry_output_root" => relpath(quarantine_root(root, entry), root),
        "files" => [Dict{String,Any}(
            "role" => file.role,
            "original_relative_path" => relpath(file.path, root),
            "quarantine_relative_path" => relpath(joinpath(quarantine_root(root, entry),
                relpath(file.path, geometry_output_root(root, entry))), root),
            "sha256" => file.sha256, "bytes" => file.bytes, "data_rows" => file.data_rows,
            "header_sha256" => file.header_sha256) for file in files],
        "terminal_geometry_count_before" => length(completed),
        "terminal_geometry_indices_before" => sort!([geometry_identity_value(entries[index])
            for index in completed]),
        "atomic_move" => "directory_rename",
        "source_files_preserved" => true,
        "prohibitions" => prohibited_actions(),
    )
end

function summary_path(root, entry)
    path = joinpath(geometry_output_root(root, entry), SUMMARY_FILE_NAME)
    isfile(path) || error("summary output is missing: $path")
    path
end

function shard_paths(root, entry)
    dir = joinpath(geometry_output_root(root, entry), SHARD_DIR_NAME)
    isdir(dir) || error("shard directory is missing: $dir")
    paths = sort!(filter(path -> endswith(path, ".csv"),
        joinpath.(dir, readdir(dir))))
    length(paths) == 1 || error("expected exactly one shard for one-geometry checkpoint")
    paths
end

function geometry_output_has_files(root, entry)
    target = geometry_output_root(root, entry)
    isdir(target) || return false
    any(!isempty(files) for (_, _, files) in walkdir(target))
end

function validate_quarantine_manifest_live(root, identity, source_manifest;
        expected_geometry_index=nothing)
    path = quarantine_manifest_path(root)
    isfile(path) && isfile(string(path, ".sha256")) ||
        error("quarantine manifest/checksum pair is incomplete")
    args = Any["--quarantine-only", path, "--output-root", root,
        "--expected-policy", identity.policy_sha256,
        "--expected-common", identity.common_digest,
        "--expected-selection", SELECTION_SHA256,
        "--expected-diff", identity.complete_diff_sha256,
        "--source-manifest-sha256", source_manifest.digest]
    expected_geometry_index === nothing ||
        append!(args, ["--expected-geometry-index", string(expected_geometry_index)])
    output = run_python_file(CHECKPOINT_VALIDATOR, args...)
    fields = split(chomp(output), '\t'; keepempty=true)
    length(fields) == 4 && fields[1] == "QUARANTINE" ||
        error("quarantine manifest validator failed")
    (; path, digest=fields[2], geometry_index=parse(Int, fields[3]),
       file_count=parse(Int, fields[4]), changed=false)
end

function rebind_quarantine_manifest_live(root, identity, source_manifest;
        expected_geometry_index=nothing)
    path = quarantine_manifest_path(root)
    args = Any["--quarantine-rebind", path,
        "--output-root", root, "--source-manifest-path", source_manifest.path,
        "--expected-policy", identity.policy_sha256,
        "--expected-common", identity.common_digest,
        "--expected-selection", SELECTION_SHA256,
        "--expected-diff", identity.complete_diff_sha256,
        "--source-manifest-sha256", source_manifest.digest]
    expected_geometry_index === nothing ||
        append!(args, ["--expected-geometry-index", string(expected_geometry_index)])
    output = run_python_file(CHECKPOINT_VALIDATOR, args...)
    fields = split(chomp(output), '\t'; keepempty=true)
    length(fields) == 5 && fields[1] == "QUARANTINE_REBOUND" ||
        error("quarantine manifest rebind validator failed")
    info = validate_quarantine_manifest_live(root, identity, source_manifest;
        expected_geometry_index=expected_geometry_index)
    info.digest == fields[2] || error("quarantine manifest rebind digest mismatch")
    (; info..., changed=fields[2] != fields[5], previous_digest=fields[5])
end

function quarantine_nonterminal_partials(root, entries, completed, identity, source_manifest)
    next_index = findfirst(index -> !(index in completed), eachindex(entries))
    next_index === nothing && error("quarantine requested but no uncheckpointed geometry remains")
    next_entry = entries[next_index]
    manifest = quarantine_manifest_path(root)
    manifest_exists = isfile(manifest) || isfile(string(manifest, ".sha256"))
    target_files = partial_output_files(root, next_entry)
    destination = quarantine_root(root, next_entry)
    if manifest_exists
        info = rebind_quarantine_manifest_live(root, identity, source_manifest)
        info.geometry_index == geometry_identity_value(next_entry) && !isempty(target_files) &&
            error("quarantine manifest exists but original partial output was not moved")
        isempty(target_files) ||
            error("uncheckpointed output exists alongside an existing quarantine manifest")
        isdir(quarantine_root(root, next_entry)) ||
            error("quarantine manifest points to a missing quarantine directory")
        for (index, entry) in enumerate(entries)
            index in completed && continue
            geometry_output_has_files(root, entry) &&
                error("uncheckpointed output exists outside the quarantined next geometry")
        end
        return info
    end
    partial_indices = Int[]
    for (index, entry) in enumerate(entries)
        index in completed && continue
        geometry_output_has_files(root, entry) && push!(partial_indices, index)
    end
    all(index -> index == next_index, partial_indices) ||
        error("uncheckpointed output exists outside the next fixed geometry")
    expected_paths = [
        joinpath(geometry_output_root(root, next_entry), SUMMARY_FILE_NAME),
        joinpath(geometry_output_root(root, next_entry), SHARD_DIR_NAME,
            geometry_shard_name(next_entry)),
    ]
    destination_files = output_files_at(destination)
    qparent = joinpath(root, QUARANTINE_DIR_NAME, QUARANTINE_VERSION)
    qparent_files = output_files_at(qparent)
    destination_expected_paths = [
        joinpath(destination, SUMMARY_FILE_NAME),
        joinpath(destination, SHARD_DIR_NAME, geometry_shard_name(next_entry)),
    ]
    all(path -> path in destination_expected_paths, qparent_files) ||
        error("quarantine directory contains unexpected files")
    if isempty(target_files) && !isempty(destination_files)
        sort!(destination_files) == sort!(copy(destination_expected_paths)) ||
            error("interrupted quarantine contains unexpected files")
        moved_files = [validate_old_header_only_csv(destination_expected_paths[1],
                PILOT_SUMMARY_FIELDS, "summary"),
            validate_old_header_only_csv(destination_expected_paths[2],
                PILOT_BRANCH_FIELDS, "shard")]
        files = [(path=expected_paths[index], sha256=moved_files[index].sha256,
            bytes=moved_files[index].bytes, data_rows=0, role=moved_files[index].role,
            header_sha256=moved_files[index].header_sha256,
            partial_policy_version=moved_files[index].partial_policy_version,
            partial_kinv_rule=moved_files[index].partial_kinv_rule,
            failure_reason=moved_files[index].failure_reason)
            for index in eachindex(expected_paths)]
    elseif isempty(target_files)
        isempty(destination_files) || error("quarantine destination is incomplete")
        return nothing
    else
        sort!(target_files) == sort!(copy(expected_paths)) ||
            error("uncheckpointed partial contains unexpected files")
        isempty(destination_files) ||
            error("quarantine destination exists alongside an original partial")
        source_files = [validate_old_header_only_csv(expected_paths[1], PILOT_SUMMARY_FIELDS, "summary"),
            validate_old_header_only_csv(expected_paths[2], PILOT_BRANCH_FIELDS, "shard")]
        mkpath(dirname(destination))
        mv(geometry_output_root(root, next_entry), destination; force=false)
        for file in source_files
            moved = joinpath(destination, relpath(file.path, geometry_output_root(root, next_entry)))
            sha256_file(moved) == file.sha256 || error("quarantine move changed file hash: $moved")
            Int64(filesize(moved)) == file.bytes || error("quarantine move changed file size: $moved")
        end
        files = source_files
    end
    payload = quarantine_manifest_fields(root, next_entry, identity, source_manifest,
        entries, completed, files)
    digest, _ = hashed_json_write(manifest, payload)
    info = validate_quarantine_manifest_live(root, identity, source_manifest;
        expected_geometry_index=geometry_identity_value(next_entry))
    info.digest == digest || error("quarantine manifest digest changed during validation")
    (; info..., changed=true)
end

function quarantine_identity_fields(info)
    info === nothing && return Dict{String,Any}(
        "quarantine_manifest_status" => "none",
        "quarantine_manifest_path" => nothing,
        "quarantine_manifest_sha256" => nothing,
        "quarantine_geometry_index" => nothing)
    Dict{String,Any}(
        "quarantine_manifest_status" => "quarantined",
        "quarantine_manifest_path" => info.path,
        "quarantine_manifest_sha256" => info.digest,
        "quarantine_geometry_index" => info.geometry_index)
end

legacy_summary_path(root) = joinpath(root, SUMMARY_FILE_NAME)

function legacy_shard_paths(root)
    dir = joinpath(root, SHARD_DIR_NAME)
    isdir(dir) || error("legacy shard directory is missing: $dir")
    paths = sort!(filter(path -> endswith(path, ".csv"), joinpath.(dir, readdir(dir))))
    length(paths) == 1 || error("expected exactly one legacy shard for geometry 1")
    paths
end

function legacy_checkpoint_path(root, entry)
    name = string("h11_", lpad(entry.h11, 3, '0'), "_np_", lpad(entry.polytope, 7, '0'),
        "_cy_", lpad(entry.frst, 7, '0'), ".checkpoint-v5.json")
    joinpath(root, CHECKPOINT_DIR_NAME, name)
end

function preserve_superseded_checkpoint(path)
    checksum_path = string(path, ".sha256")
    fields = split(strip(read(checksum_path, String)))
    length(fields) == 2 && fields[2] == basename(path) ||
        error("superseded checkpoint checksum line is malformed")
    digest = sha256_file(path)
    fields[1] == digest || error("superseded checkpoint checksum mismatch")
    preserved = string(path, ".superseded-", digest[1:16], ".json")
    atomic_copy(path, preserved)
    atomic_write(string(preserved, ".sha256"), string(digest, "  ", basename(preserved), "\n"))
    digest
end

function validate_legacy_bundle(root, entry)
    validator = CHECKPOINT_VALIDATOR
    isfile(validator) || error("checkpoint validator is missing: $validator")
    command = run_python_file(validator, "--legacy-root", root, "--summary",
        legacy_summary_path(root), "--shard", only(legacy_shard_paths(root)), "--geometry-sha256",
        entry.artifact_sha256)
    fields = split(chomp(command), '\t'; keepempty=true)
    length(fields) == 11 && fields[1] == "LEGACY" || error("legacy checkpoint validator failed")
    (; legacy_checkpoint_sha256=fields[2], summary_sha256=fields[3], shard_sha256=fields[4],
       summary_rows=parse(Int, fields[5]), branch_rows=parse(Int, fields[6]),
       max_allocated_bytes=parse(Int64, fields[7]), max_output_bytes=parse(Int64, fields[8]),
       max_estimated_stage_bytes=parse(Int64, fields[9]), max_stage_bytes=parse(Int64, fields[10]),
       summary_row_digests=split(fields[11], ','; keepempty=false),
       legacy_summary_path=legacy_summary_path(root), legacy_shard_path=only(legacy_shard_paths(root)))
end

function validate_legacy_canonical_bundle(root, entry)
    checkpoint = legacy_checkpoint_path(root, entry)
    summary = joinpath(geometry_output_root(root, entry), SUMMARY_FILE_NAME)
    shard_directory = joinpath(geometry_output_root(root, entry), SHARD_DIR_NAME)
    shard_candidates = isdir(shard_directory) ? sort!(filter(path -> endswith(path, ".csv"),
        joinpath.(shard_directory, readdir(shard_directory)))) : String[]
    length(shard_candidates) == 1 ||
        error("legacy canonical checkpoint requires exactly one per-geometry shard")
    shard = only(shard_candidates)
    output = run_python_file(CHECKPOINT_VALIDATOR, "--legacy-canonical", checkpoint,
        "--summary", summary, "--shard", shard, "--geometry-sha256", entry.artifact_sha256,
        "--expected-h11", string(entry.h11), "--expected-polytope", string(entry.polytope),
        "--expected-frst", string(entry.frst))
    fields = split(chomp(output), '\t'; keepempty=true)
    length(fields) == 11 && fields[1] == "LEGACY_CANONICAL" ||
        error("legacy canonical checkpoint validator failed")
    (; legacy_checkpoint_sha256=fields[2], summary_sha256=fields[3], shard_sha256=fields[4],
       summary_rows=parse(Int, fields[5]), branch_rows=parse(Int, fields[6]),
       max_allocated_bytes=parse(Int64, fields[7]), max_output_bytes=parse(Int64, fields[8]),
       max_estimated_stage_bytes=parse(Int64, fields[9]), max_stage_bytes=parse(Int64, fields[10]),
       summary_row_digests=isempty(fields[11]) ? String[] : split(fields[11], ','),
       legacy_summary_path=summary, legacy_shard_path=shard)
end

function validate_canonical_migration_bundle(root, entry)
    checkpoint = checkpoint_path(root, entry)
    summary = summary_path(root, entry)
    shard = only(shard_paths(root, entry))
    output = run_python_file(CHECKPOINT_VALIDATOR, "--migration-source", checkpoint,
        "--summary", summary, "--shard", shard, "--geometry-sha256",
        entry.artifact_sha256, "--expected-h11", string(entry.h11),
        "--expected-polytope", string(entry.polytope), "--expected-frst",
        string(entry.frst))
    fields = split(chomp(output), '\t'; keepempty=true)
    length(fields) == 11 && fields[1] == "MIGRATION_SOURCE" ||
        error("canonical migration source validator failed")
    (; legacy_checkpoint_sha256=fields[2], summary_sha256=fields[3],
       shard_sha256=fields[4], summary_rows=parse(Int, fields[5]),
       branch_rows=parse(Int, fields[6]), max_allocated_bytes=parse(Int64, fields[7]),
       max_output_bytes=parse(Int64, fields[8]),
       max_estimated_stage_bytes=parse(Int64, fields[9]),
       max_stage_bytes=parse(Int64, fields[10]),
       summary_row_digests=isempty(fields[11]) ? String[] : split(fields[11], ','),
       legacy_summary_path=summary, legacy_shard_path=shard)
end

function checkpoint_validate(path, identity, entry, meta, root)
    summary = summary_path(root, entry)
    shard = only(shard_paths(root, entry))
    source_manifest_path = joinpath(root, EXECUTION_SOURCE_MANIFEST_NAME)
    source_manifest = validate_live_execution_source_manifest(source_manifest_path)
    config = canonical_run_configuration(entry, meta, identity;
        execution_source_manifest_sha256=source_manifest.digest,
        execution_source_hashes=source_manifest.source_hashes)
    config_digest = run_configuration_digest(config)
    command = run_python_file(CHECKPOINT_VALIDATOR, "--canonical", path, "--summary", summary,
        "--shard", shard, "--geometry-sha256", entry.artifact_sha256,
        "--sidecar-sha256", sha256_file(meta.path), "--policy-sha256", identity.policy_sha256,
        "--common-digest", identity.common_digest, "--selection-sha256", SELECTION_SHA256,
        "--continuation-sha256", meta.continuation_sha256, "--diff-sha256", identity.complete_diff_sha256,
        "--project-sha256", PROJECT_SHA256, "--manifest-sha256", MANIFEST_SHA256,
        "--run-config-sha256", config_digest,
        "--source-manifest-sha256", source_manifest.digest,
        "--expected-h11", string(entry.h11), "--expected-polytope", string(entry.polytope),
        "--expected-frst", string(entry.frst), "--expected-geometry-index",
        string(geometry_identity_value(entry)),
        "--output-root", root)
    fields = split(chomp(command), '\t'; keepempty=true)
    length(fields) == 2 && fields[1] == "CANONICAL" || error("canonical checkpoint validator failed")
    fields[2]
end

function manifest_validate(path, identity)
    checksum = string(path, ".sha256")
    command = run_python(MANIFEST_PARSER, path, checksum, RUNNER_SCHEMA,
        identity.policy_sha256, SELECTION_SHA256)
    startswith(chomp(command), "MANIFEST\t") || error("manifest validator failed")
    sha256_file(path)
end

function identity_manifest_validate(path, schema, identity, source_manifest, root)
    isfile(path) || error("identity manifest is missing: $path")
    output = run_python_file(CHECKPOINT_VALIDATOR, "--manifest-only", path,
        "--expected-schema", schema, "--expected-policy", identity.policy_sha256,
        "--expected-common", identity.common_digest, "--expected-selection", SELECTION_SHA256,
        "--expected-diff", identity.complete_diff_sha256, "--expected-project", PROJECT_SHA256,
        "--expected-manifest", MANIFEST_SHA256, "--source-manifest-sha256", source_manifest.digest,
        "--output-root", root)
    fields = split(chomp(output), '\t'; keepempty=true)
    length(fields) == 3 && fields[1] == "MANIFEST" || error("identity manifest validator failed: $path")
    fields[2]
end

function resource_check(new_output_bytes::Int64)
    rss = Int64(Sys.maxrss())
    rss <= MAX_RSS_BYTES || error("resource cap exceeded: RSS=$(rss) > $(MAX_RSS_BYTES)")
    new_output_bytes <= MAX_NEW_OUTPUT_BYTES || error("resource cap exceeded: output=$(new_output_bytes) > $(MAX_NEW_OUTPUT_BYTES)")
    (; rss_bytes=rss, new_output_bytes)
end

function calculation_resource_evidence(root)
    output = run_python_file(CHECKPOINT_VALIDATOR, "--resource-evidence-root", root)
    fields = split(chomp(output), '\t'; keepempty=true)
    length(fields) == 6 && fields[1] == "RESOURCE_EVIDENCE" ||
        error("calculation resource-evidence validation failed")
    maxrss = parse(Int64, fields[2])
    geometry_index = parse(Int, fields[3])
    source_path = fields[4]
    source_sha256 = fields[5]
    checkpoint_count = parse(Int, fields[6])
    maxrss <= MAX_RSS_BYTES || error("preserved calculation RSS exceeds the cap")
    Dict{String,Any}(
        "maxrss_bytes" => maxrss,
        "rss_cap_bytes" => MAX_RSS_BYTES,
        "output_cap_bytes" => MAX_NEW_OUTPUT_BYTES,
        "peak_geometry_index" => geometry_index,
        "source_checkpoint_path" => source_path,
        "source_checkpoint_sha256" => source_sha256,
        "calculation_checkpoint_count" => checkpoint_count,
        "provenance" => "checksum-bound original calculation checkpoint",
    )
end

function options_for_run(run_id, identity, meta)
    Dict{Symbol,Any}(
        :scale_status => :physical, :volume_normalization => :full,
        :scale_grid => collect(SCALE_VALUES), :run_id => run_id,
        :max_branches => MAX_BRANCHES, :max_stage_allocated_bytes => MAX_STAGE_ALLOCATED_BYTES,
        :negative_mode_range => nothing, :max_negative_modes => MAX_NEGATIVE_MODES,
        :correction_tolerance => CORRECTION_TOLERANCE, :correction_iterations => CORRECTION_ITERATIONS,
        :matching_tolerance => MATCHING_TOLERANCE, :duplicate_tolerance => DUPLICATE_TOLERANCE,
        :zero_eigenvalue_tolerance => ZERO_EIGENVALUE_TOLERANCE,
        :source_identity => meta.source_identity, :configuration_digest => meta.geometry_digest,
        :phase_convention => meta.phase, :units => meta.units, :normalization => meta.normalization,
        :precision_bits => meta.precision_bits, :bracket_number => 0, :previous_scale => 1.0,
        :physical_domain_certificate => nothing, :domain_status => :missing_evidence,
        :domain_reason => "certificate is evaluated by the current physical evaluator at each scale")
end

function scale_record_from_legacy(scale_string, digest)
    Dict{String,Any}(
        "scale_string" => scale_string,
        "scale_value" => scale_string,
        "terminal_status" => "completed",
        "physical_scaling_gate_status" => "passed",
        "physical_control_gate_status" => "not_established",
        "physical_viability_status" => "not_established",
        "coverage_label" => COVERAGE_LABEL,
        "summary_row_sha256" => digest,
    )
end

function adopt_existing_geometry(entry, meta, identity, env, run_id, root, legacy, source_manifest;
        copy_outputs=true, superseded_checkpoint_sha256=nothing)
    preexisting_output_bytes = output_tree_bytes(root)
    config = canonical_run_configuration(entry, meta, identity;
        execution_source_manifest_sha256=source_manifest.digest,
        execution_source_hashes=source_manifest.source_hashes)
    config_digest = run_configuration_digest(config)
    target_root = geometry_output_root(root, entry)
    summary = joinpath(target_root, SUMMARY_FILE_NAME)
    shard = joinpath(target_root, SHARD_DIR_NAME, basename(legacy.legacy_shard_path))
    for (source, destination, expected_hash) in ((legacy.legacy_summary_path, summary, legacy.summary_sha256),
            (legacy.legacy_shard_path, shard, legacy.shard_sha256))
        if isfile(destination)
            sha256_file(destination) == expected_hash ||
                error("existing per-geometry output hash mismatch: $destination")
        elseif copy_outputs
            atomic_copy(source, destination)
        else
            error("validated migration source output is missing: $destination")
        end
    end
    summary_bytes = Int64(filesize(summary))
    shard_bytes = Int64(filesize(shard))
    observed_rss = Int64(Sys.maxrss())
    legacy_adoption = Dict{String,Any}(
        "adopted_without_recomputation" => true,
        "migration_mode" => copy_outputs ? "legacy_bundle_adoption" :
            "validated_canonical_checkpoint_migration",
        "legacy_checkpoint_sha256" => legacy.legacy_checkpoint_sha256,
        "legacy_summary_sha256" => legacy.summary_sha256,
        "legacy_shard_sha256" => legacy.shard_sha256)
    superseded_checkpoint_sha256 === nothing ||
        (legacy_adoption["superseded_checkpoint_sha256"] = superseded_checkpoint_sha256)
    checkpoint = Dict{String,Any}(
        "schema" => CHECKPOINT_SCHEMA,
        "checkpoint_version" => 6,
        "run_id" => run_id,
        "geometry_index" => geometry_identity_value(entry),
        "input" => Dict{String,Any}(
            "relative_path" => relpath(entry.path, DATA_ROOT),
            "artifact_sha256" => entry.artifact_sha256,
            "h11" => entry.h11, "polytope" => entry.polytope, "frst" => entry.frst,
            "polytope_id" => entry.polytope_id, "triangulation_id" => entry.triangulation_id,
            "orientifold_requested" => false,
        ),
        "geometry_artifact_sha256" => entry.artifact_sha256,
        "geometry_sidecar_name" => basename(meta.path),
        "geometry_sidecar_sha256" => sha256_file(meta.path),
        "policy_sha256" => identity.policy_sha256,
        "common_configuration_digest_sha256" => identity.common_digest,
        "selection_manifest_sha256" => SELECTION_SHA256,
        "continuation_source_sha256" => meta.continuation_sha256,
        "current_complete_git_diff_sha256" => identity.complete_diff_sha256,
        "project_sha256" => PROJECT_SHA256,
        "manifest_sha256" => MANIFEST_SHA256,
        "execution_source_manifest_path" => source_manifest.path,
        "execution_source_manifest_sha256" => source_manifest.digest,
        "execution_source_hashes" => source_manifest.source_hashes,
        "julia_executable" => env.julia_executable,
        "julia_version" => env.julia_version,
        "domain_certificate_version" => DOMAIN_CERTIFICATE_VERSION,
        "conversion_policy_version" => CONVERSION_POLICY_VERSION,
        "kinv_conversion_acceptance" => KINV_CONVERSION_RULE,
        "run_configuration" => config,
        "run_configuration_digest_sha256" => config_digest,
        "summary" => Dict{String,Any}(
            "path" => summary, "sha256" => legacy.summary_sha256,
            "bytes" => summary_bytes, "row_count" => legacy.summary_rows,
            "row_digests" => legacy.summary_row_digests,
        ),
        "summary_sha256" => legacy.summary_sha256,
        "shards" => Dict{String,Any}(
            "files" => [Dict{String,Any}(
                "path" => shard, "sha256" => legacy.shard_sha256,
                "bytes" => shard_bytes, "row_count" => legacy.branch_rows,
            )],
            "sha256" => legacy.shard_sha256,
        ),
        "shard_sha256" => legacy.shard_sha256,
        "resource_accounting" => Dict{String,Any}(
            "maxrss_bytes" => observed_rss,
            "preexisting_output_bytes_under_root" => preexisting_output_bytes,
            "output_bytes_under_root_before_checkpoint" => output_tree_bytes(root),
            "output_bytes_under_root_at_checkpoint" => 0,
            "new_output_bytes" => output_tree_bytes(root) - preexisting_output_bytes,
            "summary_bytes" => summary_bytes,
            "shard_bytes" => shard_bytes,
            "checkpoint_bytes" => 0,
            "summary_allocated_bytes_max" => legacy.max_allocated_bytes,
            "summary_output_bytes_max" => legacy.max_output_bytes,
            "summary_estimated_stage_allocated_bytes_max" => legacy.max_estimated_stage_bytes,
            "summary_max_stage_allocated_bytes" => legacy.max_stage_bytes,
            "rss_cap_bytes" => MAX_RSS_BYTES,
            "new_output_cap_bytes" => MAX_NEW_OUTPUT_BYTES,
            "output_cap_bytes" => MAX_NEW_OUTPUT_BYTES,
            "historical_maxrss_bytes" => nothing,
            "adoption_observed_rss_bytes" => observed_rss,
        ),
        "terminal_status" => "completed",
        "terminal_counts" => Dict{String,Any}(
            "terminal_geometry_count" => 1,
            "terminal_geometry_indices" => [geometry_identity_value(entry)],
            "terminal_scale_point_count" => legacy.summary_rows,
            "summary_row_count" => legacy.summary_rows,
            "branch_row_count" => legacy.branch_rows,
        ),
        "gates" => Dict{String,Any}(
            "physical_scaling_gate" => "passed",
            "physical_control_gate" => "not_established",
            "physical_viability_claim" => false,
        ),
        "coverage_label" => COVERAGE_LABEL,
        "scale_strings" => collect(SCALE_STRINGS),
        "max_negative_modes" => MAX_NEGATIVE_MODES,
        "prohibitions" => prohibited_actions(),
        "legacy_adoption" => legacy_adoption,
        "scale_records" => [scale_record_from_legacy(scale, digest)
            for (scale, digest) in zip(SCALE_STRINGS, legacy.summary_row_digests)],
    )
    path = checkpoint_path(root, entry)
    output_before_checkpoint = output_tree_bytes(root)
    checksum_bytes = Int64(67 + ncodeunits(basename(path)))
    predicted_checkpoint_bytes = Int64(0)
    for _ in 1:5
        checkpoint["resource_accounting"]["checkpoint_bytes"] = predicted_checkpoint_bytes
        checkpoint["resource_accounting"]["output_bytes_under_root_at_checkpoint"] =
            output_before_checkpoint + predicted_checkpoint_bytes
        predicted_json_bytes = Int64(length(codeunits(json_value(checkpoint))) + 1)
        next_checkpoint_bytes = predicted_json_bytes + checksum_bytes
        next_checkpoint_bytes == predicted_checkpoint_bytes && break
        predicted_checkpoint_bytes = next_checkpoint_bytes
    end
    checkpoint["resource_accounting"]["checkpoint_bytes"] = predicted_checkpoint_bytes
    checkpoint["resource_accounting"]["output_bytes_under_root_at_checkpoint"] =
        output_before_checkpoint + predicted_checkpoint_bytes
    resource_check(output_before_checkpoint + predicted_checkpoint_bytes)
    digest, written = hashed_json_write(path, checkpoint)
    Int64(written) == predicted_checkpoint_bytes || error("checkpoint byte accounting did not reproduce")
    resource_check(output_tree_bytes(root))
    (; digest, written, max_rss=observed_rss, evaluator_allocated_bytes=legacy.max_allocated_bytes,
       evaluator_output_bytes=legacy.max_output_bytes, summary_count=legacy.summary_rows,
       branch_count=legacy.branch_rows, run_configuration_digest=config_digest)
end

function terminal_manifest_payloads(root, entries, completed, checkpoint_hashes,
        identity, env, source_manifest, run_id, preexisting_output_bytes, stop_reason,
        run_configuration_digests=Dict{String,Any}(), quarantine_info=nothing)
    completed_indices = sort!(collect(completed))
    geometry_indices = [geometry_identity_value(entries[index]) for index in completed_indices]
    geometry_outputs = Dict{String,Any}()
    for index in completed_indices
        entry = entries[index]
        summary = summary_path(root, entry)
        shard = only(shard_paths(root, entry))
        geometry_outputs[geometry_output_name(entry)] = Dict{String,Any}(
            "geometry_index" => geometry_identity_value(entry),
            "summary_path" => summary, "summary_sha256" => sha256_file(summary),
            "shard_path" => shard, "shard_sha256" => sha256_file(shard))
    end
    nonterminal_partial_indices = Int[]
    nonterminal_partial_outputs = Dict{String,Any}()
    for (index, entry) in enumerate(entries)
        index in completed && continue
        target = geometry_output_root(root, entry)
        geometry_output_has_files(root, entry) || continue
        push!(nonterminal_partial_indices, geometry_identity_value(entry))
        files = Dict{String,Any}[]
        for (directory, _, names) in walkdir(target)
            for name in sort!(names)
                path = joinpath(directory, name)
                push!(files, Dict{String,Any}(
                    "relative_path" => relpath(path, root),
                    "sha256" => sha256_file(path),
                    "bytes" => filesize(path)))
            end
        end
        nonterminal_partial_outputs[geometry_output_name(entry)] = Dict{String,Any}(
            "geometry_index" => geometry_identity_value(entry),
            "status" => "partial_nonterminal_preserved",
            "included_in_terminal_counts" => false,
            "files" => files)
    end
    terminal_status = length(completed_indices) == length(entries) ? "completed" : "partial"
    resource_evidence = calculation_resource_evidence(root)
    env_payload = Dict{String,Any}(
        "schema" => ENVIRONMENT_SCHEMA, "run_id" => run_id,
        "environment" => runner_context_to_dict(env),
        "policy_sha256" => identity.policy_sha256,
        "common_configuration_digest_sha256" => identity.common_digest,
        "selection_manifest_sha256" => SELECTION_SHA256,
        "current_continuation_source_sha256" => source_manifest.source_hashes[
            "scripts/inflation_scale_continuation.jl"],
        "current_complete_git_diff_sha256" => identity.complete_diff_sha256,
        "project_sha256" => PROJECT_SHA256, "manifest_sha256" => MANIFEST_SHA256,
        "execution_source_manifest_path" => source_manifest.path,
        "execution_source_manifest_sha256" => source_manifest.digest,
        "execution_source_hashes" => source_manifest.source_hashes,
        "domain_certificate_version" => DOMAIN_CERTIFICATE_VERSION,
        "conversion_policy_version" => CONVERSION_POLICY_VERSION,
        "kinv_conversion_acceptance" => KINV_CONVERSION_RULE,
        "scale_strings" => collect(SCALE_STRINGS), "max_negative_modes" => MAX_NEGATIVE_MODES,
        "coverage_label" => COVERAGE_LABEL, "prohibitions" => prohibited_actions(),
        "terminal_geometry_count" => length(completed_indices),
        "terminal_geometry_indices" => geometry_indices,
        "terminal_scale_point_count" => 7 * length(completed_indices),
        "nonterminal_partial_geometry_indices" => nonterminal_partial_indices,
        "nonterminal_partial_geometry_outputs" => nonterminal_partial_outputs,
        "run_configuration_scope" => "geometry-specific digest is bound in each checkpoint")
    merge!(env_payload, quarantine_identity_fields(quarantine_info))
    env_path = joinpath(root, "environment_manifest-v1.json")
    env_digest, env_written = hashed_json_write(env_path, env_payload)
    output_before_accounting = output_tree_bytes(root)
    accounting_payload = Dict{String,Any}(
        "schema" => ACCOUNTING_SCHEMA, "run_id" => run_id, "status" => terminal_status,
        "fixed_input_count" => length(entries),
        "terminal_geometry_count" => length(completed_indices),
        "terminal_scale_point_count" => 7 * length(completed_indices),
        "terminal_geometry_indices" => geometry_indices,
        "nonterminal_partial_geometry_indices" => nonterminal_partial_indices,
        "nonterminal_partial_geometry_outputs" => nonterminal_partial_outputs,
        "checkpoint_hashes" => checkpoint_hashes, "geometry_outputs" => geometry_outputs,
        "policy_sha256" => identity.policy_sha256,
        "common_configuration_digest_sha256" => identity.common_digest,
        "selection_manifest_sha256" => SELECTION_SHA256,
        "current_continuation_source_sha256" => source_manifest.source_hashes[
            "scripts/inflation_scale_continuation.jl"],
        "current_complete_git_diff_sha256" => identity.complete_diff_sha256,
        "project_sha256" => PROJECT_SHA256, "manifest_sha256" => MANIFEST_SHA256,
        "execution_source_manifest_path" => source_manifest.path,
        "execution_source_manifest_sha256" => source_manifest.digest,
        "execution_source_hashes" => source_manifest.source_hashes,
        "julia_executable" => env.julia_executable, "julia_version" => env.julia_version,
        "domain_certificate_version" => DOMAIN_CERTIFICATE_VERSION,
        "conversion_policy_version" => CONVERSION_POLICY_VERSION,
        "kinv_conversion_acceptance" => KINV_CONVERSION_RULE,
        "run_configuration_digests" => run_configuration_digests,
        "scale_strings" => collect(SCALE_STRINGS), "max_negative_modes" => MAX_NEGATIVE_MODES,
        "coverage_label" => COVERAGE_LABEL,
        "physical_control_gate" => "not_established", "physical_viability_claim" => false,
        "production_claim" => false, "validated_candidate_claim" => false,
        "prohibitions" => prohibited_actions(), "stop_reason" => stop_reason,
        "preexisting_output_bytes_under_root" => preexisting_output_bytes,
        "output_bytes_under_root_before_accounting" => output_before_accounting,
        "new_output_bytes" => output_before_accounting - preexisting_output_bytes,
        "output_bytes_under_root" => output_before_accounting,
        "output_cap_bytes" => MAX_NEW_OUTPUT_BYTES,
        "resource_accounting" => merge(resource_evidence, Dict{String,Any}(
            "preexisting_output_bytes_under_root" => preexisting_output_bytes,
            "new_output_bytes" => output_before_accounting - preexisting_output_bytes,
            "output_bytes_under_root" => output_before_accounting)))
    merge!(accounting_payload, quarantine_identity_fields(quarantine_info))
    accounting_path = joinpath(root, "terminal_accounting_manifest-v1.json")
    accounting_digest, accounting_written = hashed_json_write(accounting_path, accounting_payload)
    output_before_run = output_tree_bytes(root)
    run_payload = Dict{String,Any}(
        "schema" => RUNNER_SCHEMA, "run_id" => run_id, "status" => terminal_status,
        "worktree" => WORKTREE, "branch" => REQUIRED_BRANCH, "head" => REQUIRED_COMMIT,
        "fixed_input_count" => length(entries),
        "terminal_geometry_count" => length(completed_indices),
        "terminal_scale_point_count" => 7 * length(completed_indices),
        "terminal_geometry_indices" => geometry_indices,
        "nonterminal_partial_geometry_indices" => nonterminal_partial_indices,
        "nonterminal_partial_geometry_outputs" => nonterminal_partial_outputs,
        "selection_manifest_sha256" => SELECTION_SHA256,
        "policy_sha256" => identity.policy_sha256,
        "common_configuration_digest_sha256" => identity.common_digest,
        "current_continuation_source_sha256" => source_manifest.source_hashes[
            "scripts/inflation_scale_continuation.jl"],
        "current_complete_git_diff_sha256" => identity.complete_diff_sha256,
        "project_sha256" => PROJECT_SHA256, "manifest_sha256" => MANIFEST_SHA256,
        "execution_source_manifest_path" => source_manifest.path,
        "execution_source_manifest_sha256" => source_manifest.digest,
        "execution_source_hashes" => source_manifest.source_hashes,
        "julia_executable" => env.julia_executable, "julia_version" => env.julia_version,
        "domain_certificate_version" => DOMAIN_CERTIFICATE_VERSION,
        "conversion_policy_version" => CONVERSION_POLICY_VERSION,
        "kinv_conversion_acceptance" => KINV_CONVERSION_RULE,
        "scale_strings" => collect(SCALE_STRINGS), "max_negative_modes" => MAX_NEGATIVE_MODES,
        "coverage_label" => COVERAGE_LABEL,
        "resource_limits" => Dict("maximum_resident_bytes" => MAX_RSS_BYTES,
            "maximum_new_output_bytes" => MAX_NEW_OUTPUT_BYTES,
            "one_geometry_at_a_time" => true, "atomic_checkpoint_after_each_geometry" => true,
            "idempotent_resume" => true),
        "physical_control_gate" => "not_established", "physical_viability_claim" => false,
        "production_claim" => false, "validated_candidate_claim" => false,
        "environment_manifest_sha256" => env_digest,
        "terminal_accounting_manifest_sha256" => accounting_digest,
        "checkpoint_hashes" => checkpoint_hashes,
        "run_configuration_digests" => run_configuration_digests,
        "prohibitions" => prohibited_actions(), "stop_reason" => stop_reason,
        "preexisting_output_bytes_under_root" => preexisting_output_bytes,
        "output_bytes_under_root_before_run_manifest" => output_before_run,
        "new_output_bytes" => output_before_run - preexisting_output_bytes,
        "output_bytes_under_root" => output_before_run,
        "output_cap_bytes" => MAX_NEW_OUTPUT_BYTES,
        "resource_accounting" => merge(resource_evidence, Dict{String,Any}(
            "preexisting_output_bytes_under_root" => preexisting_output_bytes,
            "new_output_bytes" => output_before_run - preexisting_output_bytes,
            "output_bytes_under_root" => output_before_run)))
    merge!(run_payload, quarantine_identity_fields(quarantine_info))
    run_path = joinpath(root, "run_manifest-v1.json")
    run_digest, run_written = hashed_json_write(run_path, run_payload)
    resource_check(output_tree_bytes(root))
    (; environment_manifest_sha256=env_digest,
       terminal_accounting_manifest_sha256=accounting_digest,
       run_manifest_sha256=run_digest,
       terminal_geometry_indices=geometry_indices,
       nonterminal_partial_indices,
       output_bytes_under_root=output_tree_bytes(root))
end

function geometry_output_payloads(root, entries, completed)
    outputs = Dict{String,Any}()
    for index in sort!(collect(completed))
        entry = entries[index]
        summary = summary_path(root, entry)
        shard = only(shard_paths(root, entry))
        outputs[geometry_output_name(entry)] = Dict{String,Any}(
            "geometry_index" => geometry_identity_value(entry),
            "summary_path" => summary, "summary_sha256" => sha256_file(summary),
            "shard_path" => shard, "shard_sha256" => sha256_file(shard))
    end
    outputs
end

function run_one_geometry(entry, meta, identity, run_id, output_root, new_output_bytes, source_manifest)
    preexisting_output_bytes = new_output_bytes
    geometry_root = geometry_output_root(output_root, entry)
    summary_output = joinpath(geometry_root, SUMMARY_FILE_NAME)
    shard_output = joinpath(geometry_root, SHARD_DIR_NAME,
        string("inflation_scale_continuation_shard_", lpad(entry.h11, 3, '0'), "_np_",
            lpad(entry.polytope, 7, '0'), "_cy_", lpad(entry.frst, 7, '0'), ".csv"))
    pilot_prepare_csv(summary_output, PILOT_SUMMARY_FIELDS; append=false)
    pilot_prepare_csv(shard_output, PILOT_BRANCH_FIELDS; append=false)
    loaded = load_geometry_evidence(entry, meta)
    options = options_for_run(run_id, identity, meta)
    hierarchy = CYAxiverse.generate.instanton_hierarchy_diagnostics(loaded.L)
    seed_info = pilot_collect_seeds(loaded.Q, loaded.L;
        max_branches=MAX_BRANCHES, max_negative_modes=MAX_NEGATIVE_MODES,
        max_stage_allocated_bytes=MAX_STAGE_ALLOCATED_BYTES)
    seed_info.status == :completed || error("bounded seed evaluator did not complete: $(seed_info.status)")
    previous = nothing
    previous_minima = 0
    records = Any[]
    max_rss = Int64(Sys.maxrss())
    total_allocated = Int64(0)
    total_evaluator_output = Int64(0)
    total_branch_rows = 0
    for (scale_string, scale) in zip(SCALE_STRINGS, SCALE_VALUES)
        resource_check(output_tree_bytes(output_root))
        measured = @timed _pilot_scale_records(entry === nothing ? nothing : loaded.geom,
            DATA_ROOT, entry.path, loaded.Q, loaded.L, loaded.K, hierarchy, seed_info,
            scale, previous, previous_minima, options, loaded.geometry_data)
        result = measured.value
        summary = merge(result.summary, (; wall_seconds=measured.time,
            allocated_bytes=measured.bytes, output_bytes=Base.summarysize(result.records)))
        String(summary.physical_scaling_gate_status) == "passed" || error("scale gate failed at $scale_string")
        String(summary.physical_control_gate_status) == "not_established" || error("control gate changed at $scale_string")
        String(summary.physical_viability_status) == "not_established" || error("viability status changed at $scale_string")
        pilot_append_csv(summary_output, summary, PILOT_SUMMARY_FIELDS)
        for record in result.records
            pilot_append_csv(shard_output, _pilot_branch_row(record, result.context), PILOT_BRANCH_FIELDS)
        end
        push!(records, Dict{String,Any}(
            "scale_string" => scale_string, "scale_value" => scale_string,
            "terminal_status" => "completed", "physical_scaling_gate_status" => "passed",
            "physical_control_gate_status" => "not_established",
            "physical_viability_status" => "not_established",
            "summary" => runner_context_to_dict(summary),
            "evaluator_allocated_bytes" => measured.bytes,
            "evaluator_output_bytes" => Base.summarysize(result.records),
            "wall_seconds" => measured.time, "maxrss_bytes" => Int64(Sys.maxrss())))
        previous = result.records
        previous_minima = result.minima
        max_rss = max(max_rss, Int64(Sys.maxrss()))
        total_allocated += measured.bytes
        total_evaluator_output += Base.summarysize(result.records)
        total_branch_rows += length(result.records)
        resource_check(output_tree_bytes(output_root))
    end
    config = canonical_run_configuration(entry, meta, identity;
        execution_source_manifest_sha256=source_manifest.digest,
        execution_source_hashes=source_manifest.source_hashes)
    config_digest = run_configuration_digest(config)
    summary_sha = sha256_file(summary_output)
    shard_sha = sha256_file(shard_output)
    checkpoint = Dict{String,Any}(
        "schema" => CHECKPOINT_SCHEMA, "checkpoint_version" => 6,
        "run_id" => run_id, "geometry_index" => geometry_identity_value(entry),
        "input" => Dict("relative_path" => relpath(entry.path, DATA_ROOT), "artifact_sha256" => entry.artifact_sha256,
            "h11" => entry.h11, "polytope" => entry.polytope, "frst" => entry.frst,
            "polytope_id" => entry.polytope_id, "triangulation_id" => entry.triangulation_id,
            "orientifold_requested" => false),
        "geometry_artifact_sha256" => entry.artifact_sha256,
        "geometry_sidecar_name" => basename(meta.path), "geometry_sidecar_sha256" => sha256_file(meta.path),
        "policy_sha256" => identity.policy_sha256, "common_configuration_digest_sha256" => identity.common_digest,
        "selection_manifest_sha256" => SELECTION_SHA256,
        "continuation_source_sha256" => meta.continuation_sha256,
        "current_complete_git_diff_sha256" => identity.complete_diff_sha256,
        "project_sha256" => PROJECT_SHA256, "manifest_sha256" => MANIFEST_SHA256,
        "execution_source_manifest_path" => source_manifest.path,
        "execution_source_manifest_sha256" => source_manifest.digest,
        "execution_source_hashes" => source_manifest.source_hashes,
        "julia_executable" => abspath(joinpath(Sys.BINDIR, Base.julia_exename())),
        "julia_version" => string(VERSION),
        "domain_certificate_version" => DOMAIN_CERTIFICATE_VERSION,
        "conversion_policy_version" => CONVERSION_POLICY_VERSION,
        "kinv_conversion_acceptance" => KINV_CONVERSION_RULE,
        "run_configuration" => config, "run_configuration_digest_sha256" => config_digest,
        "summary" => Dict("path" => summary_output, "sha256" => summary_sha,
            "bytes" => filesize(summary_output), "row_count" => length(records), "row_digests" => String[]),
        "summary_sha256" => summary_sha,
        "shards" => Dict("files" => [Dict("path" => shard_output, "sha256" => shard_sha,
            "bytes" => filesize(shard_output), "row_count" => total_branch_rows)],
            "sha256" => shard_sha),
        "shard_sha256" => shard_sha,
        "resource_accounting" => Dict("maxrss_bytes" => max_rss, "evaluator_allocated_bytes" => total_allocated,
            "evaluator_output_bytes" => total_evaluator_output,
            "new_output_bytes_before_checkpoint" => output_tree_bytes(output_root) - preexisting_output_bytes,
            "preexisting_output_bytes_under_root" => preexisting_output_bytes,
            "output_bytes_under_root_before_checkpoint" => output_tree_bytes(output_root),
            "output_bytes_under_root_at_checkpoint" => 0,
            "new_output_bytes" => output_tree_bytes(output_root) - preexisting_output_bytes,
            "summary_bytes" => filesize(summary_output), "shard_bytes" => filesize(shard_output),
            "checkpoint_bytes" => 0, "rss_cap_bytes" => MAX_RSS_BYTES,
            "new_output_cap_bytes" => MAX_NEW_OUTPUT_BYTES, "output_cap_bytes" => MAX_NEW_OUTPUT_BYTES),
        "terminal_status" => "completed", "terminal_counts" => Dict("terminal_geometry_count" => 1,
            "terminal_geometry_indices" => [geometry_identity_value(entry)],
            "terminal_scale_point_count" => length(records), "summary_row_count" => length(records),
            "branch_row_count" => total_branch_rows),
        "gates" => Dict("physical_scaling_gate" => "passed", "physical_control_gate" => "not_established",
            "physical_viability_claim" => false),
        "coverage_label" => COVERAGE_LABEL, "scale_strings" => collect(SCALE_STRINGS),
        "max_negative_modes" => MAX_NEGATIVE_MODES, "prohibitions" => prohibited_actions(),
        "scale_records" => records)
    path = checkpoint_path(output_root, entry)
    output_before_checkpoint = output_tree_bytes(output_root)
    checksum_bytes = Int64(67 + ncodeunits(basename(path)))
    predicted_checkpoint_bytes = Int64(0)
    for _ in 1:5
        checkpoint["resource_accounting"]["checkpoint_bytes"] = predicted_checkpoint_bytes
        checkpoint["resource_accounting"]["output_bytes_under_root_at_checkpoint"] =
            output_before_checkpoint + predicted_checkpoint_bytes
        predicted_json_bytes = Int64(length(codeunits(json_value(checkpoint))) + 1)
        next_checkpoint_bytes = predicted_json_bytes + checksum_bytes
        next_checkpoint_bytes == predicted_checkpoint_bytes && break
        predicted_checkpoint_bytes = next_checkpoint_bytes
    end
    checkpoint["resource_accounting"]["checkpoint_bytes"] = predicted_checkpoint_bytes
    checkpoint["resource_accounting"]["output_bytes_under_root_at_checkpoint"] =
        output_before_checkpoint + predicted_checkpoint_bytes
    resource_check(output_before_checkpoint + predicted_checkpoint_bytes)
    digest, written = hashed_json_write(path, checkpoint)
    Int64(written) == predicted_checkpoint_bytes || error("checkpoint byte accounting did not reproduce")
    resource_check(output_tree_bytes(output_root))
    (; digest, written, max_rss, evaluator_allocated_bytes=total_allocated,
       evaluator_output_bytes=total_evaluator_output, summary_count=length(records))
end

function run_options(args)
    options = Dict{Symbol,Any}(:resume => false, :resume_existing_only => false,
        :adopt_existing => false, :migrate_existing => false, :validate_only => false,
        :resume_selection_only => false, :quarantine_nonterminal_partials => false,
        :max_geometries => 1, :output_root => OUTPUT_ROOT_DEFAULT)
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--resume"
            options[:resume] = true
        elseif arg == "--resume-existing-only"
            options[:resume] = true
            options[:resume_existing_only] = true
        elseif arg == "--resume-selection-only"
            options[:resume] = true
            options[:resume_selection_only] = true
        elseif arg == "--quarantine-nonterminal-partials"
            options[:resume] = true
            options[:quarantine_nonterminal_partials] = true
        elseif arg == "--adopt-existing"
            options[:adopt_existing] = true
        elseif arg == "--migrate-existing"
            options[:migrate_existing] = true
        elseif arg == "--validate-only"
            options[:validate_only] = true
        elseif arg == "--max-geometries"
            i += 1; i <= length(args) || error("missing --max-geometries value")
            options[:max_geometries] = parse(Int, args[i])
        elseif arg == "--output-root"
            i += 1; i <= length(args) || error("missing --output-root value")
            options[:output_root] = abspath(expanduser(args[i]))
        elseif arg in ("--help", "-h")
            println("run_physical_scale_inflation_pilot_20260825.jl [--resume] [--resume-existing-only] [--resume-selection-only] [--quarantine-nonterminal-partials] [--adopt-existing] [--migrate-existing] [--validate-only] [--max-geometries N] [--output-root PATH]")
            exit(0)
        else
            error("unknown option: $arg")
        end
        i += 1
    end
    1 <= options[:max_geometries] <= 18 || error("max-geometries must be between 1 and 18")
    options[:adopt_existing] && options[:resume_existing_only] &&
        error("--adopt-existing and --resume-existing-only are mutually exclusive")
    options[:resume_selection_only] && (options[:adopt_existing] || options[:resume_existing_only]) &&
        error("--resume-selection-only cannot be combined with adoption or validation-only resume")
    options[:quarantine_nonterminal_partials] &&
        (options[:adopt_existing] || options[:resume_existing_only] || options[:migrate_existing]) &&
        error("--quarantine-nonterminal-partials requires the generic resume path")
    options[:migrate_existing] && (options[:adopt_existing] || options[:resume_existing_only]) &&
        error("--migrate-existing cannot be combined with adoption or validation-only resume")
    options
end

function main(args=ARGS)
    options = run_options(args)
    env = validate_environment()
    entries = fixed_inputs()
    identity = validate_sidecars()
    options[:validate_only] && begin
        println("validation_only=true")
        println("fixed_inputs=18")
        println("policy_sha256=", identity.policy_sha256)
        println("common_configuration_digest_sha256=", identity.common_digest)
        println("current_complete_git_diff_sha256=", identity.complete_diff_sha256)
        println("julia_version=", env.julia_version)
        return (; status=:validated, entries, identity, env)
    end
    root = options[:output_root]

    # Migrate every already-terminal canonical geometry. This path validates
    # each immutable v6 checkpoint and its own per-geometry CSV outputs, then
    # rewrites current-policy v6 checkpoints without loading an evaluator or
    # recalculating a scale point. No target selection occurs in migration.
    if options[:migrate_existing]
        preexisting_output_bytes = output_tree_bytes(root)
        mkpath(joinpath(root, CHECKPOINT_DIR_NAME))
        source_manifest_path = joinpath(root, EXECUTION_SOURCE_MANIFEST_NAME)
        source_manifest = if isfile(source_manifest_path)
            try
                validate_live_execution_source_manifest(source_manifest_path)
            catch stale_source_manifest_error
                # Migration is the only path allowed to rebind provenance after
                # a reviewed source repair. Preserve the checksum-bound prior
                # manifest before writing the current live-source identity.
                preserve_superseded_checkpoint(source_manifest_path)
                write_execution_source_manifest(root)
            end
        else
            write_execution_source_manifest(root)
        end
        migration_entries = NamedTuple[]
        for entry in entries
            canonical = checkpoint_path(root, entry)
            has_checkpoint = isfile(canonical) || isfile(string(canonical, ".sha256"))
            has_checkpoint && (!isfile(canonical) || !isfile(string(canonical, ".sha256"))) &&
                error("partial migrated checkpoint pair is not resumable")
            has_checkpoint && push!(migration_entries, entry)
        end
        isempty(migration_entries) && error("no existing canonical checkpoints to migrate")
        completed = Set{Int}()
        checkpoint_hashes = Dict{String,String}()
        run_configuration_digests = Dict{String,String}()
        for entry in migration_entries
            index = only(findall(item -> item.path == entry.path, entries))
            meta = geometry_sidecar(entry)
            canonical = checkpoint_path(root, entry)
            result = try
                digest = checkpoint_validate(canonical, identity, entry, meta, root)
                (; digest, run_configuration_digest=run_configuration_digest(
                    canonical_run_configuration(entry, meta, identity;
                        execution_source_manifest_sha256=source_manifest.digest,
                        execution_source_hashes=source_manifest.source_hashes)))
            catch current_checkpoint_error
                # The migration validator checks the old terminal identity and
                # output hashes without trusting stale live-source bindings.
                # Only after that proof is complete is the old checkpoint
                # preserved and rebound to the current policy/source set.
                legacy = validate_canonical_migration_bundle(root, entry)
                superseded = preserve_superseded_checkpoint(canonical)
                adopt_existing_geometry(entry, meta, identity, env,
                    "physical-scale-inflation-pilot-20260825", root, legacy, source_manifest;
                    copy_outputs=false, superseded_checkpoint_sha256=superseded)
            end
            checkpoint_hashes[basename(canonical)] = result.digest
            run_configuration_digests[geometry_output_name(entry)] =
                result.run_configuration_digest
            push!(completed, index)
        end
        quarantine_info = if isfile(quarantine_manifest_path(root)) ||
                isfile(string(quarantine_manifest_path(root), ".sha256"))
            rebind_quarantine_manifest_live(root, identity, source_manifest)
        else
            nothing
        end
        next_index = findfirst(index -> !(index in completed), eachindex(entries))
        stop_reason = next_index === nothing ? "migration stop after existing terminal geometries" :
            "bounded stop before geometry $(geometry_identity_value(entries[next_index]))"
        manifests = terminal_manifest_payloads(root, entries, completed, checkpoint_hashes,
            identity, env, source_manifest, "physical-scale-inflation-pilot-20260825",
            preexisting_output_bytes, stop_reason, run_configuration_digests, quarantine_info)
        println("migrated_geometries=", join(string.(manifests.terminal_geometry_indices), ","))
        println("migrated_without_recomputation=true")
        println("terminal_geometry_count=", length(completed))
        println("terminal_geometry_indices=", join(string.(manifests.terminal_geometry_indices), ","))
        println("terminal_scale_point_count=", 7 * length(completed))
        println("coverage_label=", COVERAGE_LABEL)
        println("max_negative_modes=", MAX_NEGATIVE_MODES)
        println("execution_source_manifest_sha256=", source_manifest.digest)
        println("environment_manifest_sha256=", manifests.environment_manifest_sha256)
        println("terminal_accounting_manifest_sha256=", manifests.terminal_accounting_manifest_sha256)
        println("run_manifest_sha256=", manifests.run_manifest_sha256)
        println("output_bytes_under_root=", manifests.output_bytes_under_root)
        return (; status=:partial, completed, checkpoint_hashes,
            run_configuration_digests, manifests)
    end

    # Adoption is a deliberately terminal operation for the explicitly
    # supplied legacy geometry. It validates the seven-row bundle and writes
    # one canonical checkpoint without calling the evaluator.
    if options[:adopt_existing] || options[:resume_existing_only]
        entry = only(filter(item -> item.h11 == 5 && item.polytope == 1 && item.frst == 1, entries))
        adoption_output_baseline = output_tree_bytes(root)
        if options[:adopt_existing] || options[:resume_existing_only]
            for (index, candidate) in enumerate(entries)
                index == 1 && continue
                other_checkpoint = checkpoint_path(root, candidate)
                (isfile(other_checkpoint) || isfile(string(other_checkpoint, ".sha256")) ||
                    geometry_output_has_files(root, candidate)) &&
                    error("existing adoption output contains an unbound non-target terminal or partial geometry")
            end
        end
        meta = geometry_sidecar(entry)
        mkpath(joinpath(root, CHECKPOINT_DIR_NAME))
        source_manifest_path = joinpath(root, EXECUTION_SOURCE_MANIFEST_NAME)
        source_manifest = if isfile(source_manifest_path)
            validate_live_execution_source_manifest(source_manifest_path)
        else
            options[:resume_existing_only] &&
                error("--resume-existing-only requires the execution-source manifest")
            write_execution_source_manifest(root)
        end
        canonical = checkpoint_path(root, entry)
        has_checkpoint = isfile(canonical) || isfile(string(canonical, ".sha256"))
        has_checkpoint && (!isfile(canonical) || !isfile(string(canonical, ".sha256"))) &&
            error("partial canonical checkpoint pair is not resumable")
        run_id = "physical-scale-inflation-pilot-20260825"
        result = if has_checkpoint
            digest = checkpoint_validate(canonical, identity, entry, meta, root)
            (; digest, written=Int64(0), max_rss=Int64(Sys.maxrss()),
               evaluator_allocated_bytes=Int64(0), evaluator_output_bytes=Int64(0),
               summary_count=7, branch_count=0,
               run_configuration_digest=run_configuration_digest(
                   canonical_run_configuration(entry, meta, identity;
                       execution_source_manifest_sha256=source_manifest.digest,
                       execution_source_hashes=source_manifest.source_hashes)))
        else
            options[:resume_existing_only] &&
                error("--resume-existing-only requires an existing canonical geometry-1 checkpoint")
            legacy = validate_legacy_bundle(root, entry)
            adopt_existing_geometry(entry, meta, identity, env, run_id, root, legacy, source_manifest)
        end
        config = canonical_run_configuration(entry, meta, identity;
            execution_source_manifest_sha256=source_manifest.digest,
            execution_source_hashes=source_manifest.source_hashes)
        config_digest = run_configuration_digest(config)
        if has_checkpoint && options[:resume_existing_only]
            identity_manifest_validate(joinpath(root, "environment_manifest-v1.json"),
                ENVIRONMENT_SCHEMA, identity, source_manifest, root)
            identity_manifest_validate(joinpath(root, "run_manifest-v1.json"),
                RUNNER_SCHEMA, identity, source_manifest, root)
            println("resumed_geometry=1")
            println("checkpoint_path=", canonical)
            println("checkpoint_sha256=", result.digest)
            println("execution_source_manifest_sha256=", source_manifest.digest)
            println("run_configuration_digest_sha256=", config_digest)
            println("terminal_geometry_count=1")
            println("terminal_geometry_indices=5001001")
            println("terminal_scale_point_count=7")
            println("coverage_label=", COVERAGE_LABEL)
            println("max_negative_modes=", MAX_NEGATIVE_MODES)
            return (; status=:partial, completed=[1], checkpoint_sha256=result.digest,
                run_configuration_digest_sha256=config_digest)
        end
        env_payload = Dict{String,Any}(
            "schema" => ENVIRONMENT_SCHEMA, "run_id" => run_id,
            "environment" => runner_context_to_dict(env),
            "policy_sha256" => identity.policy_sha256,
            "common_configuration_digest_sha256" => identity.common_digest,
            "selection_manifest_sha256" => SELECTION_SHA256,
            "current_continuation_source_sha256" => meta.continuation_sha256,
            "current_complete_git_diff_sha256" => identity.complete_diff_sha256,
            "project_sha256" => PROJECT_SHA256, "manifest_sha256" => MANIFEST_SHA256,
            "execution_source_manifest_path" => source_manifest.path,
            "execution_source_manifest_sha256" => source_manifest.digest,
            "execution_source_hashes" => source_manifest.source_hashes,
            "domain_certificate_version" => DOMAIN_CERTIFICATE_VERSION,
            "conversion_policy_version" => CONVERSION_POLICY_VERSION,
            "kinv_conversion_acceptance" => KINV_CONVERSION_RULE,
            "run_configuration_digest_sha256" => config_digest,
            "terminal_geometry_count" => 1,
            "terminal_geometry_indices" => [geometry_identity_value(entry)],
            "terminal_scale_point_count" => 7,
            "geometry_outputs" => geometry_output_payloads(root, entries, Set([1])),
            "scale_strings" => collect(SCALE_STRINGS), "max_negative_modes" => MAX_NEGATIVE_MODES,
            "coverage_label" => COVERAGE_LABEL,
            "preexisting_output_bytes_under_root" => adoption_output_baseline,
            "output_bytes_under_root_before_manifest" => output_tree_bytes(root),
            "new_output_bytes" => output_tree_bytes(root) - adoption_output_baseline,
            "output_bytes_under_root" => output_tree_bytes(root),
            "prohibitions" => prohibited_actions(),
        )
        env_path = joinpath(root, "environment_manifest-v1.json")
        env_digest, env_written = hashed_json_write(env_path, env_payload)
        accounting = Dict{String,Any}(
            "schema" => ACCOUNTING_SCHEMA, "run_id" => run_id, "status" => "partial",
            "fixed_input_count" => 18, "terminal_geometry_count" => 1,
            "terminal_scale_point_count" => 7, "terminal_geometry_indices" => [5001001],
            "checkpoint_hashes" => Dict(checkpoint_name(entry) => result.digest),
            "summary_sha256" => sha256_file(summary_path(root, entry)),
            "shard_sha256" => sha256_file(only(shard_paths(root, entry))),
            "policy_sha256" => identity.policy_sha256,
            "common_configuration_digest_sha256" => identity.common_digest,
            "selection_manifest_sha256" => SELECTION_SHA256,
            "current_continuation_source_sha256" => meta.continuation_sha256,
            "current_complete_git_diff_sha256" => identity.complete_diff_sha256,
            "project_sha256" => PROJECT_SHA256, "manifest_sha256" => MANIFEST_SHA256,
            "execution_source_manifest_path" => source_manifest.path,
            "execution_source_manifest_sha256" => source_manifest.digest,
            "execution_source_hashes" => source_manifest.source_hashes,
            "domain_certificate_version" => DOMAIN_CERTIFICATE_VERSION,
            "conversion_policy_version" => CONVERSION_POLICY_VERSION,
            "kinv_conversion_acceptance" => KINV_CONVERSION_RULE,
            "julia_executable" => env.julia_executable, "julia_version" => env.julia_version,
            "run_configuration_digest_sha256" => config_digest,
            "geometry_outputs" => Dict(geometry_output_name(entry) => Dict(
                "geometry_index" => geometry_identity_value(entry),
                "summary_path" => summary_path(root, entry),
                "summary_sha256" => sha256_file(summary_path(root, entry)),
                "shard_path" => only(shard_paths(root, entry)),
                "shard_sha256" => sha256_file(only(shard_paths(root, entry))))),
            "scale_strings" => collect(SCALE_STRINGS), "max_negative_modes" => MAX_NEGATIVE_MODES,
            "coverage_label" => COVERAGE_LABEL,
            "physical_control_gate" => "not_established", "physical_viability_claim" => false,
            "production_claim" => false, "validated_candidate_claim" => false,
            "prohibitions" => prohibited_actions(),
            "stop_reason" => "bounded stop before geometry 2",
            "preexisting_output_bytes_under_root" => adoption_output_baseline,
            "output_bytes_under_root_before_manifest" => output_tree_bytes(root),
            "new_output_bytes" => output_tree_bytes(root) - adoption_output_baseline,
            "output_bytes_under_root" => output_tree_bytes(root),
        )
        accounting_path = joinpath(root, "terminal_accounting_manifest-v1.json")
        accounting_digest, accounting_written = hashed_json_write(accounting_path, accounting)
        run_payload = Dict{String,Any}(
            "schema" => RUNNER_SCHEMA, "run_id" => run_id, "status" => "partial",
            "worktree" => WORKTREE, "branch" => REQUIRED_BRANCH, "head" => REQUIRED_COMMIT,
            "fixed_input_count" => 18, "terminal_geometry_count" => 1,
            "terminal_scale_point_count" => 7, "terminal_geometry_indices" => [5001001],
            "selection_manifest_sha256" => SELECTION_SHA256,
            "policy_sha256" => identity.policy_sha256,
            "common_configuration_digest_sha256" => identity.common_digest,
            "current_continuation_source_sha256" => meta.continuation_sha256,
            "current_complete_git_diff_sha256" => identity.complete_diff_sha256,
            "project_sha256" => PROJECT_SHA256, "manifest_sha256" => MANIFEST_SHA256,
            "execution_source_manifest_path" => source_manifest.path,
            "execution_source_manifest_sha256" => source_manifest.digest,
            "execution_source_hashes" => source_manifest.source_hashes,
            "domain_certificate_version" => DOMAIN_CERTIFICATE_VERSION,
            "conversion_policy_version" => CONVERSION_POLICY_VERSION,
            "kinv_conversion_acceptance" => KINV_CONVERSION_RULE,
            "scale_strings" => collect(SCALE_STRINGS), "max_negative_modes" => MAX_NEGATIVE_MODES,
            "coverage_label" => COVERAGE_LABEL,
            "resource_limits" => Dict("maximum_resident_bytes" => MAX_RSS_BYTES,
                "maximum_new_output_bytes" => MAX_NEW_OUTPUT_BYTES,
                "one_geometry_at_a_time" => true, "atomic_checkpoint_after_each_geometry" => true,
                "idempotent_resume" => true),
            "physical_control_gate" => "not_established", "physical_viability_claim" => false,
            "production_claim" => false, "validated_candidate_claim" => false,
            "environment_manifest_sha256" => env_digest,
            "terminal_accounting_manifest_sha256" => accounting_digest,
            "run_configuration_digest_sha256" => config_digest,
            "checkpoint_hashes" => Dict(checkpoint_name(entry) => result.digest),
            "geometry_outputs" => geometry_output_payloads(root, entries, Set([1])),
            "preexisting_output_bytes_under_root" => adoption_output_baseline,
            "output_bytes_under_root_before_run_manifest" => output_tree_bytes(root),
            "new_output_bytes" => output_tree_bytes(root) - adoption_output_baseline,
            "prohibitions" => prohibited_actions(),
            "stop_reason" => "bounded stop before geometry 2",
            "output_bytes_under_root" => output_tree_bytes(root),
        )
        run_path = joinpath(root, "run_manifest-v1.json")
        run_digest, run_written = hashed_json_write(run_path, run_payload)
        resource_check(output_tree_bytes(root))
        println("adopted_geometry=1")
        println("adopted_without_recomputation=true")
        println("checkpoint_path=", canonical)
        println("checkpoint_sha256=", result.digest)
        println("summary_sha256=", sha256_file(summary_path(root, entry)))
        println("shard_sha256=", sha256_file(only(shard_paths(root, entry))))
        println("run_configuration_digest_sha256=", config_digest)
        println("environment_manifest_sha256=", env_digest)
        println("terminal_accounting_manifest_sha256=", accounting_digest)
        println("run_manifest_sha256=", run_digest)
        println("terminal_geometry_count=1")
        println("terminal_scale_point_count=7")
        println("coverage_label=", COVERAGE_LABEL)
        println("max_negative_modes=", MAX_NEGATIVE_MODES)
        return (; status=:partial, completed=[1], checkpoint_sha256=result.digest,
            run_configuration_digest_sha256=config_digest, run_manifest_sha256=run_digest)
    end
    checkpoint_dir = joinpath(root, CHECKPOINT_DIR_NAME)
    if isdir(root) && !options[:resume] && !isempty(readdir(root))
        error("output root already exists; use --resume to preserve verified partial results")
    end
    run_output_baseline = output_tree_bytes(root)
    mkpath(checkpoint_dir)
    source_manifest_path = joinpath(root, EXECUTION_SOURCE_MANIFEST_NAME)
    source_manifest = if isfile(source_manifest_path)
        validate_live_execution_source_manifest(source_manifest_path)
    else
        options[:resume] && error("--resume requires the execution-source manifest")
        write_execution_source_manifest(root)
    end
    if options[:resume]
        identity_manifest_validate(joinpath(root, "environment_manifest-v1.json"),
            ENVIRONMENT_SCHEMA, identity, source_manifest, root)
        identity_manifest_validate(joinpath(root, "run_manifest-v1.json"),
            RUNNER_SCHEMA, identity, source_manifest, root)
    end
    checkpoint_hashes = Dict{String,String}()
    completed = Set{Int}()
    for (index, entry) in enumerate(entries)
        path = checkpoint_path(root, entry)
        meta = geometry_sidecar(entry)
        if isfile(path) || isfile(string(path, ".sha256"))
            isfile(path) && isfile(string(path, ".sha256")) || error("partial checkpoint pair is not resumable")
            checkpoint_hashes[basename(path)] = checkpoint_validate(path, identity, entry, meta, root)
            push!(completed, index)
        end
    end
    quarantine_info = if options[:quarantine_nonterminal_partials]
        quarantine_nonterminal_partials(root, entries, completed, identity, source_manifest)
    else
        for (index, entry) in enumerate(entries)
            geometry_output_has_files(root, entry) && !(index in completed) &&
                error("uncheckpointed per-geometry output refuses target selection: $(geometry_output_name(entry))")
        end
        nothing
    end
    options[:resume] || isempty(completed) || error("existing checkpoints require --resume")
    run_id = "physical-scale-inflation-pilot-20260825"
    new_output_bytes = output_tree_bytes(root)
    environment_payload = Dict{String,Any}("schema" => ENVIRONMENT_SCHEMA, "run_id" => run_id,
        "environment" => runner_context_to_dict(env), "policy_sha256" => identity.policy_sha256,
        "common_configuration_digest_sha256" => identity.common_digest, "selection_manifest_sha256" => SELECTION_SHA256,
        "current_complete_git_diff_sha256" => identity.complete_diff_sha256,
        "execution_source_manifest_path" => source_manifest.path,
        "execution_source_manifest_sha256" => source_manifest.digest,
        "execution_source_hashes" => source_manifest.source_hashes,
        "julia_executable" => env.julia_executable, "julia_version" => env.julia_version,
        "project_sha256" => PROJECT_SHA256, "manifest_sha256" => MANIFEST_SHA256,
        "domain_certificate_version" => DOMAIN_CERTIFICATE_VERSION,
        "conversion_policy_version" => CONVERSION_POLICY_VERSION,
        "kinv_conversion_acceptance" => KINV_CONVERSION_RULE,
        "terminal_geometry_count" => length(completed),
        "terminal_geometry_indices" => sort!([geometry_identity_value(entries[i]) for i in completed]),
        "terminal_scale_point_count" => 7 * length(completed),
        "nonterminal_partial_geometry_indices" => [geometry_identity_value(entries[i])
            for i in 1:length(entries) if !(i in completed) && geometry_output_has_files(root, entries[i])],
        "scale_strings" => collect(SCALE_STRINGS),
        "max_negative_modes" => MAX_NEGATIVE_MODES, "physical_control_gate" => "not_established",
        "coverage_label" => COVERAGE_LABEL,
        "preexisting_output_bytes_under_root" => run_output_baseline,
        "output_bytes_under_root_before_manifest" => run_output_baseline,
        "new_output_bytes" => output_tree_bytes(root) - run_output_baseline,
        "output_bytes_under_root" => output_tree_bytes(root),
        "prohibitions" => prohibited_actions(),
        "run_configuration_scope" => "geometry-specific digest is bound in each checkpoint")
    merge!(environment_payload, quarantine_identity_fields(quarantine_info))
    env_path = joinpath(root, "environment_manifest-v1.json")
    env_digest, env_written = hashed_json_write(env_path, environment_payload)
    new_output_bytes = output_tree_bytes(root)
    run_payload = Dict{String,Any}("schema" => RUNNER_SCHEMA, "run_id" => run_id,
        "status" => length(completed) == 18 ? "completed" : "running", "worktree" => WORKTREE,
        "branch" => REQUIRED_BRANCH, "head" => REQUIRED_COMMIT, "fixed_input_count" => 18,
        "selection_manifest_sha256" => SELECTION_SHA256, "policy_sha256" => identity.policy_sha256,
        "common_configuration_digest_sha256" => identity.common_digest,
        "current_complete_git_diff_sha256" => identity.complete_diff_sha256,
        "execution_source_manifest_path" => source_manifest.path,
        "execution_source_manifest_sha256" => source_manifest.digest,
        "execution_source_hashes" => source_manifest.source_hashes,
        "project_sha256" => PROJECT_SHA256, "manifest_sha256" => MANIFEST_SHA256,
        "domain_certificate_version" => DOMAIN_CERTIFICATE_VERSION,
        "conversion_policy_version" => CONVERSION_POLICY_VERSION,
        "kinv_conversion_acceptance" => KINV_CONVERSION_RULE,
        "scale_strings" => collect(SCALE_STRINGS), "max_negative_modes" => MAX_NEGATIVE_MODES,
        "coverage_label" => COVERAGE_LABEL,
        "max_branches" => MAX_BRANCHES, "resource_limits" => Dict("maximum_resident_bytes" => MAX_RSS_BYTES,
            "maximum_new_output_bytes" => MAX_NEW_OUTPUT_BYTES, "one_geometry_at_a_time" => true,
            "atomic_checkpoint_after_each_geometry" => true, "idempotent_resume" => true),
        "physical_control_gate" => "not_established", "physical_viability_claim" => false,
        "production_claim" => false, "validated_candidate_claim" => false,
        "environment_manifest_sha256" => env_digest,
        "checkpoint_hashes" => checkpoint_hashes, "terminal_geometry_count" => length(completed),
        "terminal_geometry_indices" => sort!([geometry_identity_value(entries[i]) for i in completed]),
        "terminal_scale_point_count" => 7 * length(completed),
        "nonterminal_partial_geometry_indices" => [geometry_identity_value(entries[i])
            for i in 1:length(entries) if !(i in completed) && geometry_output_has_files(root, entries[i])],
        "geometry_outputs" => geometry_output_payloads(root, entries, completed),
        "preexisting_output_bytes_under_root" => run_output_baseline,
        "output_bytes_under_root_before_run_manifest" => output_tree_bytes(root),
        "new_output_bytes" => output_tree_bytes(root) - run_output_baseline,
        "output_bytes_under_root" => output_tree_bytes(root),
        "prohibitions" => prohibited_actions())
    merge!(run_payload, quarantine_identity_fields(quarantine_info))
    run_path = joinpath(root, "run_manifest-v1.json")
    run_digest, run_written = hashed_json_write(run_path, run_payload)
    new_output_bytes = run_output_baseline
    resource_check(output_tree_bytes(root))
    identity_manifest_validate(joinpath(root, "environment_manifest-v1.json"),
        ENVIRONMENT_SCHEMA, identity, source_manifest, root)
    identity_manifest_validate(joinpath(root, "run_manifest-v1.json"),
        RUNNER_SCHEMA, identity, source_manifest, root)
    remaining = [index for index in eachindex(entries) if !(index in completed)]
    targets = isempty(remaining) ? Int[] : remaining[1:min(options[:max_geometries], length(remaining))]
    if options[:resume_selection_only]
        manifests = terminal_manifest_payloads(root, entries, completed, checkpoint_hashes,
            identity, env, source_manifest, run_id, run_output_baseline,
            "selection-only stop before next geometry", Dict{String,Any}(), quarantine_info)
        next_index = isempty(remaining) ? nothing : first(remaining)
        println("resume_selection_only=true")
        println("next_geometry_index=", next_index === nothing ? "none" : geometry_identity_value(entries[next_index]))
        println("runner_targets=", join(string.(targets), ","))
        println("terminal_geometry_count=", length(completed))
        println("terminal_geometry_indices=", join(string.([geometry_identity_value(entries[i]) for i in sort!(collect(completed))]), ","))
        println("terminal_scale_point_count=", 7 * length(completed))
        println("coverage_label=", COVERAGE_LABEL)
        println("max_negative_modes=", MAX_NEGATIVE_MODES)
        quarantine_info === nothing || println("quarantine_manifest_sha256=", quarantine_info.digest)
        println("environment_manifest_sha256=", manifests.environment_manifest_sha256)
        println("terminal_accounting_manifest_sha256=", manifests.terminal_accounting_manifest_sha256)
        println("run_manifest_sha256=", manifests.run_manifest_sha256)
        println("output_bytes_under_root=", manifests.output_bytes_under_root)
        return (; status=:partial, completed, checkpoint_hashes, targets, manifests)
    end
    println("runner_fixed_inputs=18")
    println("runner_targets=", join(string.(targets), ","))
    println("max_negative_modes=", MAX_NEGATIVE_MODES)
    println("scale_strings=", join(SCALE_STRINGS, ","))
    for index in targets
        resource_check(output_tree_bytes(root))
        entry = entries[index]
        meta = geometry_sidecar(entry)
        result = run_one_geometry(entry, meta, identity, run_id, root, new_output_bytes, source_manifest)
        new_output_bytes = output_tree_bytes(root)
        checkpoint_hashes[basename(checkpoint_path(root, entry))] = result.digest
        push!(completed, index)
        accounting = Dict{String,Any}("schema" => ACCOUNTING_SCHEMA, "run_id" => run_id,
            "status" => length(completed) == 18 ? "completed" : "partial",
            "fixed_input_count" => 18, "terminal_geometry_count" => length(completed),
            "terminal_scale_point_count" => 7 * length(completed),
            "terminal_geometry_indices" => sort!([geometry_identity_value(entries[i]) for i in completed]),
        "checkpoint_hashes" => checkpoint_hashes,
            "geometry_outputs" => geometry_output_payloads(root, entries, completed),
            "policy_sha256" => identity.policy_sha256,
            "common_configuration_digest_sha256" => identity.common_digest,
            "selection_manifest_sha256" => SELECTION_SHA256, "current_complete_git_diff_sha256" => identity.complete_diff_sha256,
            "execution_source_manifest_path" => source_manifest.path,
            "execution_source_manifest_sha256" => source_manifest.digest,
            "execution_source_hashes" => source_manifest.source_hashes,
            "project_sha256" => PROJECT_SHA256, "manifest_sha256" => MANIFEST_SHA256,
            "domain_certificate_version" => DOMAIN_CERTIFICATE_VERSION,
            "conversion_policy_version" => CONVERSION_POLICY_VERSION,
            "kinv_conversion_acceptance" => KINV_CONVERSION_RULE,
            "julia_executable" => env.julia_executable, "julia_version" => env.julia_version,
            "scale_strings" => collect(SCALE_STRINGS), "max_negative_modes" => MAX_NEGATIVE_MODES,
            "physical_control_gate" => "not_established", "physical_viability_claim" => false,
            "production_claim" => false, "validated_candidate_claim" => false,
            "coverage_label" => COVERAGE_LABEL, "prohibitions" => prohibited_actions(),
            "nonterminal_partial_geometry_indices" => [geometry_identity_value(entries[i])
                for i in 1:length(entries) if !(i in completed) && geometry_output_has_files(root, entries[i])],
            "resource_accounting" => Dict("maxrss_bytes" => result.max_rss,
                "preexisting_output_bytes_under_root" => run_output_baseline,
                "new_output_bytes" => output_tree_bytes(root) - run_output_baseline,
                "output_bytes_under_root_before_manifest" => output_tree_bytes(root),
                "output_bytes_under_root" => output_tree_bytes(root),
                "output_cap_bytes" => MAX_NEW_OUTPUT_BYTES,
                "new_output_cap_bytes" => MAX_NEW_OUTPUT_BYTES))
        merge!(accounting, quarantine_identity_fields(quarantine_info))
        accounting_path = joinpath(root, "terminal_accounting_manifest-v1.json")
        accounting_digest, accounting_written = hashed_json_write(accounting_path, accounting)
        new_output_bytes = output_tree_bytes(root)
        run_payload["status"] = length(completed) == 18 ? "completed" : "partial"
        run_payload["checkpoint_hashes"] = checkpoint_hashes
        run_payload["terminal_geometry_count"] = length(completed)
        run_payload["terminal_geometry_indices"] = sort!([geometry_identity_value(entries[i]) for i in completed])
        run_payload["terminal_scale_point_count"] = 7 * length(completed)
        run_payload["geometry_outputs"] = geometry_output_payloads(root, entries, completed)
        run_payload["preexisting_output_bytes_under_root"] = run_output_baseline
        run_payload["output_bytes_under_root_before_run_manifest"] = output_tree_bytes(root)
        run_payload["new_output_bytes"] = output_tree_bytes(root) - run_output_baseline
        run_payload["output_bytes_under_root"] = output_tree_bytes(root)
        run_digest, run_written = hashed_json_write(run_path, run_payload)
        new_output_bytes = output_tree_bytes(root)
        resource_check(new_output_bytes)
        println("checkpoint_geometry=", index, " checkpoint_sha256=", result.digest,
            " terminal_geometry_count=", length(completed), " terminal_scale_point_count=", 7 * length(completed))
        println("checkpoint_maxrss_bytes=", result.max_rss, " new_output_bytes=", new_output_bytes)
        GC.gc()
    end
    println("terminal_geometry_count=", length(completed))
    println("terminal_scale_point_count=", 7 * length(completed))
    println("physical_control_gate=not_established")
    println("physical_viability_claim=false")
    println("production_qualified=false")
    println("validated=false")
    (; status=length(completed) == 18 ? :completed : :partial, completed, checkpoint_hashes,
       environment_manifest_sha256=env_digest, run_manifest_sha256=run_digest, new_output_bytes)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
