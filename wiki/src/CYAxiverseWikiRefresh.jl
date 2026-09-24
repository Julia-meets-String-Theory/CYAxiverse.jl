module CYAxiverseWikiRefresh

using Dates
using HTTP
using JSON3
using SHA
using YAML

export main, Drift, material, StateValidationError, BrokenSourceError,
       _validate_manifest, _validate_state, _validate_page_key,
       _snapshot_digest, _validate_packet, _atomic_text_write,
       _locked_atomic_json_write, _file_digest, _packet_object,
       _state_page_digest,
       EXIT_CLEAN, EXIT_DRIFT, EXIT_BROKEN_SOURCE, EXIT_OWNER_REVIEW

const EXIT_CLEAN = 0
const EXIT_DRIFT = 10
const EXIT_BROKEN_SOURCE = 11
const EXIT_OWNER_REVIEW = 12

const GITHUB_API = "https://api.github.com"
const SHA40_RE = r"^[0-9a-f]{40}$"

struct RemoteError <: Exception
    status::Int
    url::String
    body::String
end

struct BrokenSourceError <: Exception
    message::String
end

struct StateValidationError <: Exception
    message::String
end

Base.showerror(io::IO, e::RemoteError) =
    print(io, "remote request failed: HTTP $(e.status) $(e.url)\n$(e.body)")
Base.showerror(io::IO, e::BrokenSourceError) = print(io, e.message)
Base.showerror(io::IO, e::StateValidationError) = print(io, e.message)

struct GitHubClient
    owner::String
    repo::String
    token::Union{Nothing,String}
end

struct Drift
    key::String
    title::String
    authority::String
    mode::String
    verified_commit::String
    current_head::String
    head_changed::Bool
    changed_sources::Vector{Dict{String,Any}}
    issue_changes::Vector{Dict{String,Any}}
    pr_changes::Vector{Dict{String,Any}}
end

material(d::Drift) =
    d.head_changed || !isempty(d.changed_sources) ||
    !isempty(d.issue_changes) || !isempty(d.pr_changes)

function _plain(x)
    if x isa AbstractDict
        Dict{String,Any}(string(k) => _plain(v) for (k, v) in pairs(x))
    elseif x isa AbstractVector
        Any[_plain(v) for v in x]
    else
        x
    end
end

function _require_exact_keys(object, required::Tuple, context::AbstractString;
        optional::Tuple=())
    object isa AbstractDict ||
        throw(StateValidationError("$context must be an object"))
    actual = String[]
    for key in keys(object)
        key isa AbstractString || throw(StateValidationError(
            "$context keys must be strings"))
        push!(actual, String(key))
    end
    actual_keys = Set(actual)
    required_keys = Set(String.(required))
    allowed_keys = union(required_keys, Set(String.(optional)))
    missing = sort!(collect(setdiff(required_keys, actual_keys)))
    unexpected = sort!(collect(setdiff(actual_keys, allowed_keys)))
    isempty(missing) && isempty(unexpected) || throw(StateValidationError(
        "$context keys are invalid (missing: $(join(missing, ", ")); " *
        "unexpected: $(join(unexpected, ", ")))"))
    true
end

function _json_read(text::AbstractString)
    isempty(strip(text)) && return Dict{String,Any}()
    try
        _plain(JSON3.read(text))
    catch error
        throw(StateValidationError("invalid JSON: $(sprint(showerror, error))"))
    end
end

function _load_manifest(path::AbstractString)
    isfile(path) || throw(StateValidationError("manifest not found: $path"))
    data = try
        _plain(YAML.load_file(path))
    catch error
        throw(StateValidationError("invalid manifest YAML: $(sprint(showerror, error))"))
    end
    data isa AbstractDict ||
        throw(StateValidationError("manifest root must be an object"))
    Dict{String,Any}(String(k) => v for (k, v) in pairs(data))
end

function _file_digest(path::AbstractString)
    isfile(path) || return nothing
    bytes2hex(SHA.sha256(read(path)))
end

function _load_state(path::AbstractString)
    isfile(path) || return nothing, nothing
    digest = _file_digest(path)
    state = open(path, "r") do io
        _json_read(read(io, String))
    end
    state, digest
end

function _with_state_lock(f::Function, path::AbstractString)
    mkpath(dirname(path))
    lockdir = string(path, ".lock")
    try
        mkdir(lockdir)
    catch error
        isdir(lockdir) && throw(StateValidationError(
            "state lock is already held: $lockdir"))
        rethrow(error)
    end
    try
        return f()
    finally
        isdir(lockdir) && rm(lockdir; recursive=true, force=true)
    end
end

function _atomic_text_write(path::AbstractString, text::AbstractString)
    mkpath(dirname(path))
    tmp = tempname(dirname(path))
    try
        open(tmp, "w") do io
            write(io, text)
            flush(io)
        end
        mv(tmp, path; force=true)
    finally
        isfile(tmp) && rm(tmp; force=true)
    end
    nothing
end

function _locked_atomic_text_write(path::AbstractString, text::AbstractString;
        expected_digest::Union{Nothing,String}=nothing,
        require_absent::Bool=false,
        before_replace::Union{Nothing,Function}=nothing)
    _with_state_lock(path) do
        if require_absent && isfile(path)
            throw(StateValidationError("refusing write: $path already exists"))
        end
        if expected_digest !== nothing
            _file_digest(path) == expected_digest || throw(StateValidationError(
                "refusing write: $path changed before lock acquisition"))
        end

        tmp = tempname(dirname(path))
        try
            open(tmp, "w") do io
                write(io, text)
                flush(io)
            end
            before_replace === nothing || before_replace()

            # Re-check while the interprocess lock is still held. Every supported
            # mutating invocation uses this lock, making compare + replace one
            # serialized transaction.
            if expected_digest !== nothing
                _file_digest(path) == expected_digest || throw(StateValidationError(
                    "refusing write: $path changed during locked transaction"))
            elseif require_absent && isfile(path)
                throw(StateValidationError(
                    "refusing write: $path appeared during locked transaction"))
            end

            mv(tmp, path; force=true)
        finally
            isfile(tmp) && rm(tmp; force=true)
        end
        nothing
    end
end

function _locked_atomic_json_write(path::AbstractString, object;
        expected_digest::Union{Nothing,String}=nothing,
        require_absent::Bool=false,
        before_replace::Union{Nothing,Function}=nothing)
    _locked_atomic_text_write(path, string(JSON3.write(object), "\n");
        expected_digest=expected_digest,
        require_absent=require_absent,
        before_replace=before_replace)
end

function _request_json(method::AbstractString, url::AbstractString,
        headers::Vector{Pair{String,String}})
    last_response = nothing
    for attempt in 1:3
        response = HTTP.request(method, url, headers;
            status_exception=false, readtimeout=30)
        last_response = response
        if 200 <= response.status < 300
            text = String(response.body)
            return isempty(strip(text)) ? Dict{String,Any}() : _json_read(text)
        end
        if response.status == 429 || 500 <= response.status < 600
            attempt < 3 && sleep(2.0^(attempt - 1))
            continue
        end
        throw(RemoteError(response.status, String(url), String(response.body)))
    end
    response = last_response
    throw(RemoteError(response.status, String(url), String(response.body)))
end

function _github_headers(client::GitHubClient)
    headers = Pair{String,String}[
        "Accept" => "application/vnd.github+json",
        "X-GitHub-Api-Version" => "2022-11-28",
        "User-Agent" => "cyaxiverse-wiki-refresh/0.1",
    ]
    client.token === nothing ||
        push!(headers, "Authorization" => "Bearer $(client.token)")
    headers
end

_repo_base(client::GitHubClient) =
    "$GITHUB_API/repos/$(client.owner)/$(client.repo)"

_github_get(client::GitHubClient, path::AbstractString) =
    _request_json("GET", "$(_repo_base(client))/$path", _github_headers(client))

function _commit_info(client::GitHubClient, ref::AbstractString)
    data = _github_get(client, "commits/$ref")
    String(data["sha"]), String(data["commit"]["tree"]["sha"])
end

function _tree_blobs(client::GitHubClient, ref::AbstractString)
    commit_sha, tree_sha = _commit_info(client, ref)
    data = _github_get(client, "git/trees/$tree_sha?recursive=1")
    get(data, "truncated", false) && throw(BrokenSourceError(
        "GitHub returned a truncated repository tree for $ref"))
    blobs = Dict{String,String}()
    for entry in data["tree"]
        get(entry, "type", "") == "blob" || continue
        blobs[String(entry["path"])] = String(entry["sha"])
    end
    commit_sha, blobs
end

function _issue_snapshot(client::GitHubClient, number::Integer)
    data = _github_get(client, "issues/$number")
    Dict{String,Any}(
        "number" => Int(number),
        "title" => String(get(data, "title", "")),
        "state" => String(get(data, "state", "")),
        "state_reason" => get(data, "state_reason", nothing),
        "closed_at" => get(data, "closed_at", nothing),
    )
end

function _pr_snapshot(client::GitHubClient, number::Integer)
    data = _github_get(client, "pulls/$number")
    Dict{String,Any}(
        "number" => Int(number),
        "title" => String(get(data, "title", "")),
        "state" => String(get(data, "state", "")),
        "draft" => Bool(get(data, "draft", false)),
        "merged_at" => get(data, "merged_at", nothing),
        "head_sha" => String(data["head"]["sha"]),
        "base_ref" => String(data["base"]["ref"]),
    )
end

function _valid_sha(value)
    value isa AbstractString && occursin(SHA40_RE, String(value))
end

function _validate_manifest(manifest)
    manifest isa AbstractDict ||
        throw(StateValidationError("manifest root must be an object"))
    try
        haskey(manifest, "notion") &&
            throw(StateValidationError(
                "public manifest must not contain Notion API configuration"))
        _require_exact_keys(manifest,
            ("schema_version", "repository", "initial_verified_commit", "pages"),
            "manifest")
        Int(get(manifest, "schema_version", 0)) == 1 ||
            throw(StateValidationError("unsupported manifest schema_version"))
        repository = get(manifest, "repository", nothing)
        repository isa AbstractDict ||
            throw(StateValidationError("manifest.repository is required"))
        _require_exact_keys(repository, ("owner", "name", "branch"),
            "manifest.repository")
        for key in ("owner", "name", "branch")
            haskey(repository, key) ||
                throw(StateValidationError("repository.$key is required"))
            isempty(strip(String(repository[key]))) &&
                throw(StateValidationError("repository.$key must not be empty"))
        end
        initial = get(manifest, "initial_verified_commit", nothing)
        _valid_sha(initial) ||
            throw(StateValidationError("initial_verified_commit must be a 40-hex SHA"))
        pages = get(manifest, "pages", nothing)
        pages isa AbstractDict && !isempty(pages) ||
            throw(StateValidationError("manifest.pages must be a non-empty mapping"))
        for (key, page) in pages
            page isa AbstractDict ||
                throw(StateValidationError("$key must be a mapping"))
            haskey(page, "notion_page_id") &&
                throw(StateValidationError(
                    "$key must not publish a private notion_page_id"))
            _require_exact_keys(page,
                ("title", "authority", "mode", "sources", "issues", "pull_requests"),
                "manifest.pages.$key"; optional=("track_head",))
            for required in (
                    "title", "authority", "mode", "sources",
                    "issues", "pull_requests")
                haskey(page, required) ||
                    throw(StateValidationError("$key.$required is required"))
            end
            String(page["mode"]) in ("semantic", "mechanical") ||
                throw(StateValidationError(
                    "$key.mode must be semantic or mechanical"))
            page["sources"] isa AbstractVector ||
                throw(StateValidationError("$key.sources must be a list"))
            page["issues"] isa AbstractVector ||
                throw(StateValidationError("$key.issues must be a list"))
            page["pull_requests"] isa AbstractVector ||
                throw(StateValidationError("$key.pull_requests must be a list"))
            String(page["title"])
            String(page["authority"])
            String.(page["sources"])
            Int.(page["issues"])
            Int.(page["pull_requests"])
            if haskey(page, "track_head") && !(page["track_head"] isa Bool)
                throw(StateValidationError(
                    "$key.track_head must be a boolean"))
            end
        end
        true
    catch error
        error isa StateValidationError && rethrow()
        throw(StateValidationError(
            "invalid manifest structure: $(sprint(showerror, error))"))
    end
end

function _validate_issue_snapshot(snapshot, expected_number::Int, context::String)
    snapshot isa AbstractDict ||
        throw(StateValidationError("$context must be an object"))
    _require_exact_keys(snapshot,
        ("number", "title", "state", "state_reason", "closed_at"), context)
    Int(get(snapshot, "number", -1)) == expected_number ||
        throw(StateValidationError("$context number mismatch"))
    for field in ("title", "state", "state_reason", "closed_at")
        haskey(snapshot, field) || throw(StateValidationError("$context.$field is required"))
    end
    true
end

function _validate_pr_snapshot(snapshot, expected_number::Int, context::String)
    snapshot isa AbstractDict ||
        throw(StateValidationError("$context must be an object"))
    _require_exact_keys(snapshot,
        ("number", "title", "state", "draft", "merged_at", "head_sha", "base_ref"),
        context)
    Int(get(snapshot, "number", -1)) == expected_number ||
        throw(StateValidationError("$context number mismatch"))
    for field in ("title", "state", "draft", "merged_at", "head_sha", "base_ref")
        haskey(snapshot, field) || throw(StateValidationError("$context.$field is required"))
    end
    _valid_sha(String(snapshot["head_sha"])) ||
        throw(StateValidationError("$context.head_sha must be a 40-hex SHA"))
    true
end

function _validate_state(manifest::Dict{String,Any}, state)
    state isa AbstractDict ||
        throw(StateValidationError("wiki state is missing"))
    try
        Int(get(state, "schema_version", 0)) == 1 ||
            throw(StateValidationError("unsupported state schema_version"))
        repository = get(state, "repository", nothing)
        repository isa AbstractDict ||
            throw(StateValidationError("state.repository is required"))
        String(get(repository, "branch", "")) ==
            String(manifest["repository"]["branch"]) ||
            throw(StateValidationError("state branch does not match manifest"))
        _valid_sha(get(repository, "head", nothing)) ||
            throw(StateValidationError(
                "state.repository.head must be a 40-hex SHA"))
        state_pages = get(state, "pages", nothing)
        state_pages isa AbstractDict ||
            throw(StateValidationError("state.pages is required"))
        manifest_keys = Set(String.(collect(keys(manifest["pages"]))))
        state_keys = Set(String.(collect(keys(state_pages))))
        manifest_keys == state_keys ||
            throw(StateValidationError(
                "state page keys do not exactly match manifest page keys"))
        for key in sort!(collect(manifest_keys))
            page = state_pages[key]
            page isa AbstractDict ||
                throw(StateValidationError(
                    "state.pages.$key must be an object"))
            _valid_sha(get(page, "verified_commit", nothing)) ||
                throw(StateValidationError(
                    "state.pages.$key.verified_commit must be a 40-hex SHA"))
            haskey(page, "reconciled_at") ||
                throw(StateValidationError(
                    "state.pages.$key.reconciled_at is required"))
            String(page["reconciled_at"])
            issues = get(page, "issues", nothing)
            prs = get(page, "pull_requests", nothing)
            issues isa AbstractDict ||
                throw(StateValidationError(
                    "state.pages.$key.issues must be an object"))
            prs isa AbstractDict ||
                throw(StateValidationError(
                    "state.pages.$key.pull_requests must be an object"))
            expected_issues =
                Set(string(Int(x)) for x in manifest["pages"][key]["issues"])
            expected_prs =
                Set(string(Int(x)) for x in manifest["pages"][key]["pull_requests"])
            Set(String.(collect(keys(issues)))) == expected_issues ||
                throw(StateValidationError(
                    "state.pages.$key Issue keys do not match manifest"))
            Set(String.(collect(keys(prs)))) == expected_prs ||
                throw(StateValidationError(
                    "state.pages.$key PR keys do not match manifest"))
            for number in expected_issues
                _validate_issue_snapshot(
                    issues[number], parse(Int, number),
                    "state.pages.$key.issues.$number")
            end
            for number in expected_prs
                _validate_pr_snapshot(
                    prs[number], parse(Int, number),
                    "state.pages.$key.pull_requests.$number")
            end
            if haskey(page, "accepted_packet_id")
                occursin(
                    r"^[0-9a-f]{64}$",
                    String(page["accepted_packet_id"])) ||
                    throw(StateValidationError(
                        "state.pages.$key.accepted_packet_id must be SHA-256 hex"))
            end
        end
        true
    catch error
        error isa StateValidationError && rethrow()
        throw(StateValidationError(
            "invalid state structure: $(sprint(showerror, error))"))
    end
end

function _validate_page_key(manifest::Dict{String,Any}, key::AbstractString)
    haskey(manifest["pages"], String(key)) ||
        throw(StateValidationError("unknown wiki page key: $key"))
    String(key)
end

function _issue_tuple(snapshot; context="Issue snapshot")
    _require_exact_keys(snapshot,
        ("number", "title", "state", "state_reason", "closed_at"), context)
    (
        number = Int(snapshot["number"]),
        title = String(snapshot["title"]),
        state = String(snapshot["state"]),
        state_reason = snapshot["state_reason"],
        closed_at = snapshot["closed_at"],
    )
end

function _pr_tuple(snapshot; context="PR snapshot")
    _require_exact_keys(snapshot,
        ("number", "title", "state", "draft", "merged_at", "head_sha", "base_ref"),
        context)
    (
        number = Int(snapshot["number"]),
        title = String(snapshot["title"]),
        state = String(snapshot["state"]),
        draft = Bool(snapshot["draft"]),
        merged_at = snapshot["merged_at"],
        head_sha = String(snapshot["head_sha"]),
        base_ref = String(snapshot["base_ref"]),
    )
end

function _snapshot_payload(manifest::Dict{String,Any}, key::String,
        page::Dict{String,Any}, head::String, tree::Dict{String,String},
        github, issue_cache::Dict{Int,Dict{String,Any}},
        pr_cache::Dict{Int,Dict{String,Any}})
    sources = NamedTuple[]
    for path in sort(String.(page["sources"]))
        haskey(tree, path) || throw(BrokenSourceError("tracked source missing at $head: $path"))
        push!(sources, (path=path, blob=tree[path]))
    end
    issues = NamedTuple[]
    for number in sort(Int.(page["issues"]))
        snap = get!(issue_cache, number) do
            _issue_snapshot(github, number)
        end
        push!(issues, _issue_tuple(snap))
    end
    prs = NamedTuple[]
    for number in sort(Int.(page["pull_requests"]))
        snap = get!(pr_cache, number) do
            _pr_snapshot(github, number)
        end
        push!(prs, _pr_tuple(snap))
    end
    repo = manifest["repository"]
    (
        schema_version = 1,
        repository = (
            owner = String(repo["owner"]),
            name = String(repo["name"]),
            branch = String(repo["branch"]),
        ),
        page = (
            key = key,
            title = String(page["title"]),
            authority = String(page["authority"]),
        ),
        current_head = head,
        sources = sources,
        issues = issues,
        pull_requests = prs,
    )
end

function _snapshot_from_dict(snapshot)
    snapshot isa AbstractDict ||
        throw(StateValidationError("packet snapshot must be an object"))
    try
        _require_exact_keys(snapshot,
            ("schema_version", "repository", "page", "current_head", "sources",
                "issues", "pull_requests"),
            "packet snapshot")
        repository = snapshot["repository"]
        page = snapshot["page"]
        repository isa AbstractDict ||
            throw(StateValidationError("packet snapshot.repository must be an object"))
        page isa AbstractDict ||
            throw(StateValidationError("packet snapshot.page must be an object"))
        _require_exact_keys(repository, ("owner", "name", "branch"),
            "packet snapshot.repository")
        _require_exact_keys(page, ("key", "title", "authority"),
            "packet snapshot.page")
        sources_raw = snapshot["sources"]
        issues_raw = snapshot["issues"]
        prs_raw = snapshot["pull_requests"]
        sources_raw isa AbstractVector ||
            throw(StateValidationError("packet snapshot.sources must be a list"))
        issues_raw isa AbstractVector ||
            throw(StateValidationError("packet snapshot.issues must be a list"))
        prs_raw isa AbstractVector ||
            throw(StateValidationError("packet snapshot.pull_requests must be a list"))

        sources = sort([
            begin
                _require_exact_keys(item, ("path", "blob"),
                    "packet snapshot source")
                (path=String(item["path"]), blob=String(item["blob"]))
            end
            for item in sources_raw
        ], by=x -> x.path)
        issues = sort([
            _issue_tuple(item) for item in issues_raw
        ], by=x -> x.number)
        prs = sort([
            _pr_tuple(item) for item in prs_raw
        ], by=x -> x.number)

        snapshot_nt = (
            schema_version = Int(snapshot["schema_version"]),
            repository = (
                owner = String(repository["owner"]),
                name = String(repository["name"]),
                branch = String(repository["branch"]),
            ),
            page = (
                key = String(page["key"]),
                title = String(page["title"]),
                authority = String(page["authority"]),
            ),
            current_head = String(snapshot["current_head"]),
            sources = sources,
            issues = issues,
            pull_requests = prs,
        )
        snapshot_nt.schema_version == 1 ||
            throw(StateValidationError("unsupported packet snapshot schema_version"))
        _valid_sha(snapshot_nt.current_head) ||
            throw(StateValidationError("packet snapshot current_head must be a 40-hex SHA"))
        snapshot_nt
    catch error
        error isa StateValidationError && rethrow()
        throw(StateValidationError(
            "invalid packet snapshot structure: $(sprint(showerror, error))"))
    end
end

_snapshot_digest(snapshot) =
    bytes2hex(SHA.sha256(Vector{UInt8}(codeunits(JSON3.write(snapshot)))))

function _canonical_change_sources(changes)
    changes isa AbstractVector ||
        throw(StateValidationError("packet changes.sources must be a list"))
    sort([
        begin
            _require_exact_keys(item, ("path", "before_blob", "after_blob"),
                "packet source change")
            (
                path=String(item["path"]),
                before_blob=item["before_blob"],
                after_blob=item["after_blob"],
            )
        end for item in changes
    ], by=x -> x.path)
end

function _canonical_issue_changes(changes)
    changes isa AbstractVector ||
        throw(StateValidationError("packet changes.issues must be a list"))
    sort([
        begin
            _require_exact_keys(item, ("number", "before", "after"),
                "packet Issue change")
            (
                number=Int(item["number"]),
                before=item["before"] === nothing ? nothing : _issue_tuple(item["before"]),
                after=_issue_tuple(item["after"]),
            )
        end for item in changes
    ], by=x -> x.number)
end

function _canonical_pr_changes(changes)
    changes isa AbstractVector ||
        throw(StateValidationError("packet changes.pull_requests must be a list"))
    sort([
        begin
            _require_exact_keys(item, ("number", "before", "after"),
                "packet PR change")
            (
                number=Int(item["number"]),
                before=item["before"] === nothing ? nothing : _pr_tuple(item["before"]),
                after=_pr_tuple(item["after"]),
            )
        end for item in changes
    ], by=x -> x.number)
end

function _state_page_baseline(page)
    page isa AbstractDict ||
        throw(StateValidationError("state page baseline must be an object"))
    try
        issues = get(page, "issues", nothing)
        prs = get(page, "pull_requests", nothing)
        issues isa AbstractDict ||
            throw(StateValidationError("state page baseline issues must be an object"))
        prs isa AbstractDict ||
            throw(StateValidationError("state page baseline pull_requests must be an object"))
        (
            verified_commit = String(page["verified_commit"]),
            issues = sort([
                _issue_tuple(item) for (_, item) in pairs(issues)
            ], by=x -> x.number),
            pull_requests = sort([
                _pr_tuple(item) for (_, item) in pairs(prs)
            ], by=x -> x.number),
        )
    catch error
        error isa StateValidationError && rethrow()
        throw(StateValidationError(
            "invalid state page baseline: $(sprint(showerror, error))"))
    end
end

_state_page_digest(page) =
    bytes2hex(SHA.sha256(Vector{UInt8}(codeunits(
        JSON3.write(_state_page_baseline(page))))))

const PACKET_EXECUTION_SURFACE =
    "Work/ChatGPT with connected Notion integration"

const PACKET_INSTRUCTIONS = [
    "Read the authoritative changed sources before editing Notion.",
    "Locate exactly one page with the packet title under the CYAxiverse wiki hierarchy.",
    "If zero or multiple exact-title matches exist, stop for resolution.",
    "Update only the affected page.",
    "Preserve superseded history where relevant.",
    "Do not infer owner decisions or scientific acceptance.",
    "Do not write to GitHub.",
    "Verify the Notion edit and explicitly attest that this exact packet_id was reconciled.",
]

function _packet_payload(drift::Drift, snapshot, baseline_state_digest::String)
    occursin(r"^[0-9a-f]{64}$", baseline_state_digest) ||
        throw(StateValidationError("baseline_state_digest must be SHA-256 hex"))
    (
        schema_version = 1,
        kind = "cyaxiverse-wiki-reconciliation",
        snapshot = snapshot,
        baseline_verified_commit = drift.verified_commit,
        baseline_state_digest = baseline_state_digest,
        changes = (
            head_changed = drift.head_changed,
            sources = _canonical_change_sources(drift.changed_sources),
            issues = _canonical_issue_changes(drift.issue_changes),
            pull_requests = _canonical_pr_changes(drift.pr_changes),
        ),
        execution_surface = PACKET_EXECUTION_SURFACE,
        instructions = copy(PACKET_INSTRUCTIONS),
    )
end

_packet_digest(payload) =
    bytes2hex(SHA.sha256(Vector{UInt8}(codeunits(JSON3.write(payload)))))

function _packet_object(drift::Drift, snapshot, baseline_page)
    payload = _packet_payload(drift, snapshot, _state_page_digest(baseline_page))
    merge((packet_id=_packet_digest(payload),), payload)
end

function _packet_payload_from_dict(packet)
    packet isa AbstractDict ||
        throw(StateValidationError("packet root must be an object"))
    try
        _require_exact_keys(packet,
            ("packet_id", "schema_version", "kind", "snapshot",
                "baseline_verified_commit", "baseline_state_digest", "changes",
                "execution_surface", "instructions"),
            "packet")
        changes = packet["changes"]
        changes isa AbstractDict ||
            throw(StateValidationError("packet changes must be an object"))
        _require_exact_keys(changes,
            ("head_changed", "sources", "issues", "pull_requests"),
            "packet changes")
        instructions = packet["instructions"]
        instructions isa AbstractVector ||
            throw(StateValidationError("packet instructions must be a list"))
        payload = (
            schema_version = Int(packet["schema_version"]),
            kind = String(packet["kind"]),
            snapshot = _snapshot_from_dict(packet["snapshot"]),
            baseline_verified_commit = String(packet["baseline_verified_commit"]),
            baseline_state_digest = String(packet["baseline_state_digest"]),
            changes = (
                head_changed = Bool(changes["head_changed"]),
                sources = _canonical_change_sources(changes["sources"]),
                issues = _canonical_issue_changes(changes["issues"]),
                pull_requests = _canonical_pr_changes(changes["pull_requests"]),
            ),
            execution_surface = String(packet["execution_surface"]),
            instructions = String.(instructions),
        )
        payload.schema_version == 1 ||
            throw(StateValidationError("unsupported packet schema_version"))
        payload.kind == "cyaxiverse-wiki-reconciliation" ||
            throw(StateValidationError("invalid packet kind"))
        _valid_sha(payload.baseline_verified_commit) ||
            throw(StateValidationError(
                "packet baseline_verified_commit must be a 40-hex SHA"))
        occursin(r"^[0-9a-f]{64}$", payload.baseline_state_digest) ||
            throw(StateValidationError(
                "packet baseline_state_digest must be SHA-256 hex"))
        payload
    catch error
        error isa StateValidationError && rethrow()
        throw(StateValidationError(
            "invalid packet structure: $(sprint(showerror, error))"))
    end
end

function _validate_packet(packet)
    payload = _packet_payload_from_dict(packet)
    packet_id = try
        String(packet["packet_id"])
    catch error
        throw(StateValidationError(
            "packet_id is required: $(sprint(showerror, error))"))
    end
    occursin(r"^[0-9a-f]{64}$", packet_id) ||
        throw(StateValidationError("packet_id must be SHA-256 hex"))
    _packet_digest(payload) == packet_id ||
        throw(StateValidationError("packet digest does not match full reconciliation payload"))
    packet_id, payload.snapshot, payload
end

function _load_packet(path::AbstractString)
    isfile(path) || throw(StateValidationError("packet not found: $path"))
    packet = open(path, "r") do io
        _json_read(read(io, String))
    end
    packet_id, snapshot, payload = _validate_packet(packet)
    packet, packet_id, snapshot, payload
end

function _snapshot_page(page::Dict{String,Any}, head::String,
        github::GitHubClient, issue_cache::Dict{Int,Dict{String,Any}},
        pr_cache::Dict{Int,Dict{String,Any}})
    issues = Dict{String,Any}()
    for raw in page["issues"]
        number = Int(raw)
        issues[string(number)] = get!(issue_cache, number) do
            _issue_snapshot(github, number)
        end
    end
    prs = Dict{String,Any}()
    for raw in page["pull_requests"]
        number = Int(raw)
        prs[string(number)] = get!(pr_cache, number) do
            _pr_snapshot(github, number)
        end
    end
    Dict{String,Any}(
        "verified_commit" => head,
        "reconciled_at" => string(now(UTC)),
        "issues" => issues,
        "pull_requests" => prs,
    )
end

function _state_page_from_snapshot(snapshot, packet_id::String)
    issues = Dict{String,Any}()
    for item in snapshot.issues
        issues[string(item.number)] = Dict{String,Any}(
            "number" => item.number,
            "title" => item.title,
            "state" => item.state,
            "state_reason" => item.state_reason,
            "closed_at" => item.closed_at,
        )
    end
    prs = Dict{String,Any}()
    for item in snapshot.pull_requests
        prs[string(item.number)] = Dict{String,Any}(
            "number" => item.number,
            "title" => item.title,
            "state" => item.state,
            "draft" => item.draft,
            "merged_at" => item.merged_at,
            "head_sha" => item.head_sha,
            "base_ref" => item.base_ref,
        )
    end
    Dict{String,Any}(
        "verified_commit" => snapshot.current_head,
        "reconciled_at" => string(now(UTC)),
        "accepted_packet_id" => packet_id,
        "issues" => issues,
        "pull_requests" => prs,
    )
end

function _bootstrap_state(manifest::Dict{String,Any}, github::GitHubClient)
    branch = String(manifest["repository"]["branch"])
    head, _ = _tree_blobs(github, branch)
    expected = String(manifest["initial_verified_commit"])
    head == expected || throw(StateValidationError(
        "refusing bootstrap: current $branch head is $head, expected $expected"))
    issue_cache = Dict{Int,Dict{String,Any}}()
    pr_cache = Dict{Int,Dict{String,Any}}()
    pages = Dict{String,Any}()
    for (key, page) in manifest["pages"]
        pages[key] = _snapshot_page(page, head, github, issue_cache, pr_cache)
    end
    Dict{String,Any}(
        "schema_version" => 1,
        "repository" => Dict("head" => head, "branch" => branch),
        "created_at" => string(now(UTC)),
        "pages" => pages,
    )
end

function _collect_drifts(manifest::Dict{String,Any}, state::Dict{String,Any},
        github; only_page::Union{Nothing,String}=nothing)
    only_page === nothing || _validate_page_key(manifest, only_page)
    branch = String(manifest["repository"]["branch"])
    current_head, current_tree = _tree_blobs(github, branch)
    baseline_trees = Dict{String,Dict{String,String}}()
    issue_cache = Dict{Int,Dict{String,Any}}()
    pr_cache = Dict{Int,Dict{String,Any}}()
    state_pages = state["pages"]
    drifts = Drift[]
    broken_sources = Dict{String,Any}[]
    snapshots = Dict{String,Any}()

    for (key, page) in manifest["pages"]
        only_page !== nothing && key != only_page && continue
        baseline = state_pages[key]
        verified_commit = String(baseline["verified_commit"])
        baseline_tree = get!(baseline_trees, verified_commit) do
            _, tree = _tree_blobs(github, verified_commit)
            tree
        end

        # Fetch every tracked Issue/PR exactly once into shared caches, then
        # derive both the actionable snapshot and the display change summary
        # from those same objects.
        for raw in page["issues"]
            number = Int(raw)
            get!(issue_cache, number) do
                _issue_snapshot(github, number)
            end
        end
        for raw in page["pull_requests"]
            number = Int(raw)
            get!(pr_cache, number) do
                _pr_snapshot(github, number)
            end
        end

        snapshot = _snapshot_payload(manifest, key, page, current_head,
            current_tree, github, issue_cache, pr_cache)
        snapshots[key] = snapshot

        changed_sources = Dict{String,Any}[]
        current_sources = Dict(item.path => item.blob for item in snapshot.sources)
        for rawpath in page["sources"]
            path = String(rawpath)
            before = get(baseline_tree, path, nothing)
            after = get(current_sources, path, nothing)
            if after === nothing
                push!(broken_sources, Dict("page" => key, "path" => path))
            elseif before != after
                push!(changed_sources, Dict(
                    "path" => path, "before_blob" => before, "after_blob" => after))
            end
        end

        issue_changes = Dict{String,Any}[]
        current_issues = Dict(string(item.number) => item for item in snapshot.issues)
        for raw in page["issues"]
            number = Int(raw)
            old = baseline["issues"][string(number)]
            current = current_issues[string(number)]
            current_dict = Dict{String,Any}(
                "number" => current.number,
                "title" => current.title,
                "state" => current.state,
                "state_reason" => current.state_reason,
                "closed_at" => current.closed_at,
            )
            old == current_dict || push!(issue_changes,
                Dict("number" => number, "before" => old, "after" => current_dict))
        end

        pr_changes = Dict{String,Any}[]
        current_prs = Dict(string(item.number) => item for item in snapshot.pull_requests)
        for raw in page["pull_requests"]
            number = Int(raw)
            old = baseline["pull_requests"][string(number)]
            current = current_prs[string(number)]
            current_dict = Dict{String,Any}(
                "number" => current.number,
                "title" => current.title,
                "state" => current.state,
                "draft" => current.draft,
                "merged_at" => current.merged_at,
                "head_sha" => current.head_sha,
                "base_ref" => current.base_ref,
            )
            old == current_dict || push!(pr_changes,
                Dict("number" => number, "before" => old, "after" => current_dict))
        end

        head_changed = Bool(get(page, "track_head", false)) &&
            verified_commit != current_head
        drift = Drift(key, String(page["title"]), String(page["authority"]),
            String(page["mode"]), verified_commit, current_head, head_changed,
            changed_sources, issue_changes, pr_changes)
        material(drift) && push!(drifts, drift)
    end
    current_head, current_tree, drifts, broken_sources, snapshots
end

_short(sha::AbstractString) = first(String(sha), min(7, length(sha)))

function _print_drift(d::Drift)
    println()
    println("MATERIAL DRIFT  ", d.key)
    println("  page: ", d.title)
    println("  verified: ", _short(d.verified_commit), " -> ", _short(d.current_head))
    d.head_changed && println("  * tracked branch head changed")
    for source in d.changed_sources
        before = source["before_blob"]
        println("  * source: ", source["path"], "  ",
            before === nothing ? "<absent>" : _short(String(before)),
            " -> ", _short(String(source["after_blob"])))
    end
    for item in d.issue_changes
        before = item["before"]; after = item["after"]
        println("  * issue #", item["number"], ": ",
            get(before, "state", "?"), " -> ", get(after, "state", "?"))
    end
    for item in d.pr_changes
        before = item["before"]; after = item["after"]
        println("  * PR #", item["number"], ": ",
            get(before, "state", "?"), " -> ", get(after, "state", "?"),
            get(after, "merged_at", nothing) === nothing ? "" : " (merged)")
    end
end

function _write_packet(directory::AbstractString, drift::Drift, snapshot, baseline_page)
    packet = _packet_object(drift, snapshot, baseline_page)
    short_id = first(packet.packet_id, 12)
    path = joinpath(directory, "$(drift.key)-$(short_id).json")
    _atomic_text_write(path, string(JSON3.write(packet), "\n"))
    path, packet.packet_id
end

function _verify_remote(manifest::Dict{String,Any}, state,
        github)
    _validate_state(manifest, state)
    branch = String(manifest["repository"]["branch"])
    head, tree = _tree_blobs(github, branch)
    failures = String[]
    for (key, page) in manifest["pages"]
        for rawpath in page["sources"]
            path = String(rawpath)
            haskey(tree, path) || push!(failures,
                "$key: tracked source missing at $branch: $path")
        end
    end
    println("Verified repository head: ", head)
    failures
end

_default_manifest() = normpath(joinpath(@__DIR__, "..", "manifest.yaml"))
_default_state() = normpath(joinpath(@__DIR__, "..", "state.json"))
_default_packet_dir() = normpath(joinpath(@__DIR__, "..", ".wiki-refresh", "packets"))

function _parse_options(args::Vector{String})
    options = Dict{String,Any}(
        "manifest" => _default_manifest(),
        "state" => _default_state(),
        "packet_dir" => _default_packet_dir(),
        "page" => nothing,
        "accept" => nothing,
        "packet" => nothing,
        "reconciled_commit" => nothing,
        "attest_notion_reconciled" => false,
        "bootstrap" => false,
    )
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--bootstrap"
            options["bootstrap"] = true
            i += 1
        elseif arg == "--attest-notion-reconciled"
            options["attest_notion_reconciled"] = true
            i += 1
        elseif arg in ("--manifest", "--state", "--packet-dir", "--page",
                "--accept", "--packet", "--reconciled-commit")
            i == length(args) && throw(StateValidationError("$arg requires a value"))
            options[replace(arg[3:end], "-" => "_")] = args[i + 1]
            i += 2
        else
            throw(StateValidationError("unknown argument: $arg"))
        end
    end
    options
end

function _client(manifest::Dict{String,Any})
    repo = manifest["repository"]
    token = get(ENV, "GITHUB_TOKEN", nothing)
    GitHubClient(String(repo["owner"]), String(repo["name"]), token)
end

function _command_check(manifest, state, github; only_page=nothing)
    state === nothing && throw(StateValidationError("wiki state is missing"))
    _validate_state(manifest, state)
    only_page === nothing || _validate_page_key(manifest, only_page)
    head, _, drifts, broken, _ = _collect_drifts(manifest, state, github; only_page)
    println("CYAxiverse Wiki Reconciliation")
    println("Current vmm head: ", head)
    if !isempty(broken)
        println("\nBROKEN CANONICAL SOURCES")
        for item in broken
            println("  ", item["page"], ": ", item["path"])
        end
        return EXIT_BROKEN_SOURCE
    end
    isempty(drifts) && begin
        println("\nNO_MATERIAL_WIKI_DRIFT")
        return EXIT_CLEAN
    end
    foreach(_print_drift, drifts)
    println("\n", length(drifts), " page(s) require reconciliation.")
    EXIT_DRIFT
end

function _command_verify(manifest, state, github)
    state === nothing && throw(StateValidationError("wiki state is missing"))
    failures = _verify_remote(manifest, state, github)
    isempty(failures) && begin
        println("VERIFY: PASS")
        return EXIT_CLEAN
    end
    println("\nVERIFY: FAIL")
    foreach(x -> println("  * ", x), failures)
    EXIT_BROKEN_SOURCE
end

function _accept_page!(manifest, state, state_digest, github,
        state_path::AbstractString, key::String, packet_path::String,
        reconciled_commit::String, attested::Bool)
    attested || throw(StateValidationError(
        "--accept requires --attest-notion-reconciled"))
    _validate_page_key(manifest, key)
    _, packet_id, packet_snapshot, packet_payload = _load_packet(packet_path)

    packet_snapshot.page.key == key ||
        throw(StateValidationError("packet page key does not match --accept"))
    packet_payload.baseline_verified_commit ==
        String(state["pages"][key]["verified_commit"]) ||
        throw(StateValidationError(
            "packet baseline commit no longer matches current wiki state"))
    packet_payload.baseline_state_digest ==
        _state_page_digest(state["pages"][key]) ||
        throw(StateValidationError(
            "packet baseline state no longer matches current wiki state"))

    repo = manifest["repository"]
    packet_snapshot.repository.owner == String(repo["owner"]) &&
    packet_snapshot.repository.name == String(repo["name"]) &&
    packet_snapshot.repository.branch == String(repo["branch"]) ||
        throw(StateValidationError("packet repository identity does not match manifest"))
    packet_snapshot.current_head == reconciled_commit ||
        throw(StateValidationError("packet head does not match --reconciled-commit"))

    branch = String(repo["branch"])
    head, tree = _tree_blobs(github, branch)
    head == reconciled_commit || throw(StateValidationError(
        "refusing acceptance: reconciled commit $reconciled_commit " *
        "does not equal current $branch head $head"))

    page = manifest["pages"][key]
    issue_cache = Dict{Int,Dict{String,Any}}()
    pr_cache = Dict{Int,Dict{String,Any}}()
    live_snapshot = _snapshot_payload(manifest, key, page, head, tree,
        github, issue_cache, pr_cache)
    _snapshot_digest(live_snapshot) == _snapshot_digest(packet_snapshot) ||
        throw(StateValidationError(
            "GitHub state changed since reconciliation packet; " *
            "generate and reconcile a fresh packet"))

    state["pages"][key] = _state_page_from_snapshot(packet_snapshot, packet_id)
    state["repository"] = Dict("head" => head, "branch" => branch)
    state["updated_at"] = string(now(UTC))
    _locked_atomic_json_write(state_path, state; expected_digest=state_digest)
    println("Accepted exact reconciliation packet ", packet_id)
    println("Page: ", key, "  vmm: ", head)
    println("No GitHub or Notion writes were made by this command.")
    EXIT_CLEAN
end

function _command_reconcile(manifest, state, state_digest, github, options)
    state_path = String(options["state"])
    if Bool(options["bootstrap"])
        state === nothing ||
            throw(StateValidationError("state already exists; refusing bootstrap"))
        bootstrapped = _bootstrap_state(manifest, github)
        _locked_atomic_json_write(state_path, bootstrapped; require_absent=true)
        println("Created initial wiki state at ", state_path)
        println("No GitHub or Notion writes were made.")
        return EXIT_CLEAN
    end

    accept = options["accept"]
    if accept !== nothing
        state === nothing && throw(StateValidationError("cannot --accept without state"))
        _validate_state(manifest, state)
        packet = options["packet"]
        reconciled = options["reconciled_commit"]
        packet === nothing && throw(StateValidationError("--accept requires --packet"))
        reconciled === nothing &&
            throw(StateValidationError("--accept requires --reconciled-commit"))
        return _accept_page!(manifest, state, state_digest, github, state_path,
            String(accept), String(packet), String(reconciled),
            Bool(options["attest_notion_reconciled"]))
    end

    state === nothing && throw(StateValidationError("wiki state is missing"))
    _validate_state(manifest, state)
    only_page = options["page"]
    only_page === nothing || _validate_page_key(manifest, String(only_page))
    _, _, drifts, broken, snapshots = _collect_drifts(manifest, state, github;
        only_page = only_page === nothing ? nothing : String(only_page))
    !isempty(broken) && return EXIT_BROKEN_SOURCE
    isempty(drifts) && begin
        println("NO_MATERIAL_WIKI_DRIFT")
        return EXIT_CLEAN
    end

    count = 0
    for drift in drifts
        _print_drift(drift)
        count += 1
        path, packet_id = _write_packet(String(options["packet_dir"]),
            drift, snapshots[drift.key], state["pages"][drift.key])
        println("  reconciliation packet: ", path)
        println("  packet_id: ", packet_id)
    end
    println("\n", count, " page(s) require Work/Notion reconciliation.")
    EXIT_OWNER_REVIEW
end

function main(args=ARGS)
    try
        argv = collect(String, args)
        command = isempty(argv) ? "check" : popfirst!(argv)
        options = _parse_options(argv)
        manifest = _load_manifest(String(options["manifest"]))
        _validate_manifest(manifest)
        state, state_digest = _load_state(String(options["state"]))
        github = _client(manifest)

        command == "check" && return _command_check(manifest, state, github;
            only_page = options["page"] === nothing ? nothing : String(options["page"]))
        command == "verify" && return _command_verify(manifest, state, github)
        command == "reconcile" && return _command_reconcile(
            manifest, state, state_digest, github, options)
        println(stderr, "Unknown command: $command")
        return 2
    catch error
        println(stderr, "wiki-refresh error: ", sprint(showerror, error))
        error isa RemoteError && return EXIT_BROKEN_SOURCE
        error isa BrokenSourceError && return EXIT_BROKEN_SOURCE
        error isa StateValidationError && return EXIT_OWNER_REVIEW
        error isa ArgumentError && return EXIT_OWNER_REVIEW
        error isa SystemError && return EXIT_OWNER_REVIEW
        error isa Base.IOError && return EXIT_OWNER_REVIEW
        rethrow()
    end
end

end
