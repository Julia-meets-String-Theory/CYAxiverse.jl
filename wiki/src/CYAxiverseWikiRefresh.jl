module CYAxiverseWikiRefresh

using Dates
using HTTP
using JSON3
using YAML

export main, Drift, material,
       EXIT_CLEAN, EXIT_DRIFT, EXIT_BROKEN_SOURCE, EXIT_OWNER_REVIEW

const EXIT_CLEAN = 0
const EXIT_DRIFT = 10
const EXIT_BROKEN_SOURCE = 11
const EXIT_OWNER_REVIEW = 12

const GITHUB_API = "https://api.github.com"
const NOTION_API = "https://api.notion.com"

struct RemoteError <: Exception
    status::Int
    url::String
    body::String
end

Base.showerror(io::IO, e::RemoteError) =
    print(io, "remote request failed: HTTP $(e.status) $(e.url)\n$(e.body)")

struct GitHubClient
    owner::String
    repo::String
    token::Union{Nothing,String}
end

struct NotionClient
    token::String
    api_version::String
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

_json_read(text::AbstractString) =
    isempty(strip(text)) ? Dict{String,Any}() : _plain(JSON3.read(text))

function _json_write_file(path::AbstractString, object)
    mkpath(dirname(path))
    open(path, "w") do io
        JSON3.write(io, object)
        write(io, '\n')
    end
end

function _load_manifest(path::AbstractString)
    isfile(path) || throw(ArgumentError("manifest not found: $path"))
    _plain(YAML.load_file(path))
end

function _load_state(path::AbstractString)
    isfile(path) || return nothing
    open(path, "r") do io
        _json_read(read(io, String))
    end
end

function _request_json(method::AbstractString, url::AbstractString,
        headers::Vector{Pair{String,String}}; body=nothing)
    response = if body === nothing
        HTTP.request(method, url, headers; status_exception=false, readtimeout=30)
    else
        HTTP.request(method, url, headers, JSON3.write(body);
            status_exception=false, readtimeout=30)
    end
    text = String(response.body)
    200 <= response.status < 300 || throw(RemoteError(response.status, String(url), text))
    isempty(strip(text)) ? Dict{String,Any}() : _json_read(text)
end

function _github_headers(client::GitHubClient)
    headers = Pair{String,String}[
        "Accept" => "application/vnd.github+json",
        "X-GitHub-Api-Version" => "2022-11-28",
        "User-Agent" => "cyaxiverse-wiki-refresh/0.1",
    ]
    client.token === nothing || push!(headers, "Authorization" => "Bearer $(client.token)")
    headers
end

_notion_headers(client::NotionClient) = Pair{String,String}[
    "Authorization" => "Bearer $(client.token)",
    "Content-Type" => "application/json",
    "Notion-Version" => client.api_version,
    "User-Agent" => "cyaxiverse-wiki-refresh/0.1",
]

_repo_base(client::GitHubClient) = "$GITHUB_API/repos/$(client.owner)/$(client.repo)"

_github_get(client::GitHubClient, path::AbstractString) =
    _request_json("GET", "$(_repo_base(client))/$path", _github_headers(client))

_notion_get(client::NotionClient, path::AbstractString) =
    _request_json("GET", "$NOTION_API/v1/$path", _notion_headers(client))

_notion_patch(client::NotionClient, path::AbstractString, body) =
    _request_json("PATCH", "$NOTION_API/v1/$path", _notion_headers(client); body)

function _commit_info(client::GitHubClient, ref::AbstractString)
    data = _github_get(client, "commits/$ref")
    String(data["sha"]), String(data["commit"]["tree"]["sha"])
end

function _tree_blobs(client::GitHubClient, ref::AbstractString)
    commit_sha, tree_sha = _commit_info(client, ref)
    data = _github_get(client, "git/trees/$tree_sha?recursive=1")
    get(data, "truncated", false) && error("GitHub returned a truncated repository tree for $ref")
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

_notion_page(client::NotionClient, page_id::AbstractString) =
    _notion_get(client, "pages/$page_id")

function _notion_markdown(client::NotionClient, page_id::AbstractString)
    data = _notion_get(client, "pages/$page_id/markdown")
    Bool(get(data, "truncated", false)) &&
        error("Notion markdown for page $page_id is truncated")
    String(data["markdown"])
end

function _notion_title(data::Dict{String,Any})
    for (_, property) in get(data, "properties", Dict{String,Any}())
        get(property, "type", "") == "title" || continue
        return join(String(get(item, "plain_text", "")) for item in get(property, "title", Any[]))
    end
    ""
end

const VERIFIED_RE = Regex("\\*\\*Verified against:\\*\\*\\s*`vmm@[0-9a-fA-F]{7,40}`")

function _verified_marker(markdown::AbstractString)
    result = match(VERIFIED_RE, markdown)
    result === nothing ? nothing : result.match
end

function _update_verified_marker!(client::NotionClient, page_id::AbstractString,
        new_commit::AbstractString)
    markdown = _notion_markdown(client, page_id)
    old = _verified_marker(markdown)
    old === nothing && return false
    new = "**Verified against:** `vmm@$new_commit`"
    body = Dict("type" => "update_content",
        "update_content" => Dict("content_updates" => Any[
            Dict("old_str" => old, "new_str" => new),
        ]))
    _notion_patch(client, "pages/$page_id/markdown", body)
    occursin(new, _notion_markdown(client, page_id)) ||
        error("Notion verification marker did not persist for page $page_id")
    true
end

function _validate_manifest(manifest::Dict{String,Any})
    Int(get(manifest, "schema_version", 0)) == 1 ||
        throw(ArgumentError("unsupported manifest schema_version"))
    repository = manifest["repository"]
    for key in ("owner", "name", "branch")
        haskey(repository, key) || throw(ArgumentError("repository.$key is required"))
    end
    haskey(manifest, "initial_verified_commit") ||
        throw(ArgumentError("initial_verified_commit is required"))
    for (key, page) in manifest["pages"]
        haskey(page, "notion_page_id") &&
            throw(ArgumentError("$key must not publish a private notion_page_id"))
        for required in ("title", "authority", "mode", "sources", "issues", "pull_requests")
            haskey(page, required) || throw(ArgumentError("$key.$required is required"))
        end
        String(page["mode"]) in ("semantic", "mechanical") ||
            throw(ArgumentError("$key.mode must be semantic or mechanical"))
    end
    true
end

function _load_page_map(path::AbstractString)
    raw = strip(get(ENV, "NOTION_PAGE_MAP_JSON", ""))
    if !isempty(raw)
        data = _json_read(raw)
        pages = haskey(data, "pages") ? data["pages"] : data
        return Dict{String,String}(String(k) => String(v) for (k, v) in pairs(pages))
    end
    isfile(path) || return Dict{String,String}()
    data = _plain(YAML.load_file(path))
    pages = get(data, "pages", Dict{String,Any}())
    Dict{String,String}(String(k) => String(v) for (k, v) in pairs(pages))
end

function _snapshot_page(page::Dict{String,Any}, head::String, github::GitHubClient,
        issue_cache::Dict{Int,Dict{String,Any}}, pr_cache::Dict{Int,Dict{String,Any}})
    issues = Dict{String,Any}()
    for raw in get(page, "issues", Any[])
        number = Int(raw)
        issues[string(number)] = get!(issue_cache, number) do
            _issue_snapshot(github, number)
        end
    end
    prs = Dict{String,Any}()
    for raw in get(page, "pull_requests", Any[])
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

function _bootstrap_state(manifest::Dict{String,Any}, github::GitHubClient)
    branch = String(manifest["repository"]["branch"])
    head, _ = _tree_blobs(github, branch)
    expected = String(manifest["initial_verified_commit"])
    head == expected || throw(ArgumentError(
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
        github::GitHubClient; only_page::Union{Nothing,String}=nothing)
    branch = String(manifest["repository"]["branch"])
    current_head, current_tree = _tree_blobs(github, branch)
    baseline_trees = Dict{String,Dict{String,String}}()
    issue_cache = Dict{Int,Dict{String,Any}}()
    pr_cache = Dict{Int,Dict{String,Any}}()
    state_pages = get(state, "pages", Dict{String,Any}())
    drifts = Drift[]
    broken_sources = Dict{String,Any}[]

    for (key, page) in manifest["pages"]
        only_page !== nothing && key != only_page && continue
        haskey(state_pages, key) ||
            throw(ArgumentError("page '$key' has no baseline state"))
        baseline = state_pages[key]
        verified_commit = String(baseline["verified_commit"])
        baseline_tree = get!(baseline_trees, verified_commit) do
            _, tree = _tree_blobs(github, verified_commit)
            tree
        end
        changed_sources = Dict{String,Any}[]
        for rawpath in get(page, "sources", Any[])
            path = String(rawpath)
            before = get(baseline_tree, path, nothing)
            after = get(current_tree, path, nothing)
            if after === nothing
                push!(broken_sources, Dict("page" => key, "path" => path))
            elseif before != after
                push!(changed_sources, Dict(
                    "path" => path, "before_blob" => before, "after_blob" => after))
            end
        end

        issue_changes = Dict{String,Any}[]
        old_issues = get(baseline, "issues", Dict{String,Any}())
        for raw in get(page, "issues", Any[])
            number = Int(raw)
            current = get!(issue_cache, number) do
                _issue_snapshot(github, number)
            end
            old = get(old_issues, string(number), nothing)
            old == current || push!(issue_changes,
                Dict("number" => number, "before" => old, "after" => current))
        end

        pr_changes = Dict{String,Any}[]
        old_prs = get(baseline, "pull_requests", Dict{String,Any}())
        for raw in get(page, "pull_requests", Any[])
            number = Int(raw)
            current = get!(pr_cache, number) do
                _pr_snapshot(github, number)
            end
            old = get(old_prs, string(number), nothing)
            old == current || push!(pr_changes,
                Dict("number" => number, "before" => old, "after" => current))
        end

        head_changed = Bool(get(page, "track_head", false)) &&
            verified_commit != current_head
        drift = Drift(key, String(page["title"]), String(page["authority"]),
            String(page["mode"]), verified_commit, current_head, head_changed,
            changed_sources, issue_changes, pr_changes)
        material(drift) && push!(drifts, drift)
    end
    current_head, drifts, broken_sources
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
            before === nothing ? "<unbaselined>" : get(before, "state", "?"),
            " -> ", get(after, "state", "?"))
    end
    for item in d.pr_changes
        before = item["before"]; after = item["after"]
        println("  * PR #", item["number"], ": ",
            before === nothing ? "<unbaselined>" : get(before, "state", "?"),
            " -> ", get(after, "state", "?"),
            get(after, "merged_at", nothing) === nothing ? "" : " (merged)")
    end
end

function _write_packet(directory::AbstractString, drift::Drift)
    mkpath(directory)
    path = joinpath(directory, "$(drift.key).json")
    packet = Dict{String,Any}(
        "schema_version" => 1,
        "kind" => "cyaxiverse-wiki-reconciliation",
        "page_key" => drift.key,
        "page_title" => drift.title,
        "authority" => drift.authority,
        "verified_commit" => drift.verified_commit,
        "current_head" => drift.current_head,
        "head_changed" => drift.head_changed,
        "changed_sources" => drift.changed_sources,
        "issue_changes" => drift.issue_changes,
        "pull_request_changes" => drift.pr_changes,
        "instructions" => [
            "Read the authoritative changed sources before editing Notion.",
            "Update only the affected page.",
            "Preserve superseded history where relevant.",
            "Do not infer owner decisions or scientific acceptance.",
            "Do not write to GitHub.",
            "After semantic reconciliation, run reconcile --accept $(drift.key).",
        ],
    )
    _json_write_file(path, packet)
    path
end

function _verify_remote(manifest::Dict{String,Any}, github::GitHubClient,
        notion::Union{Nothing,NotionClient}, page_map::Dict{String,String})
    branch = String(manifest["repository"]["branch"])
    head, tree = _tree_blobs(github, branch)
    failures = String[]
    for (key, page) in manifest["pages"]
        for rawpath in get(page, "sources", Any[])
            path = String(rawpath)
            haskey(tree, path) || push!(failures,
                "$key: tracked source missing at $branch: $path")
        end
        notion === nothing && continue
        haskey(page_map, key) || begin
            push!(failures, "$key: private Notion page mapping is missing")
            continue
        end
        try
            remote = _notion_page(notion, page_map[key])
            title = _notion_title(remote)
            expected = String(page["title"])
            !isempty(title) && title != expected &&
                push!(failures, "$key: Notion title '$title' != '$expected'")
            _notion_markdown(notion, page_map[key])
        catch error
            push!(failures, "$key: Notion verification failed: $(sprint(showerror, error))")
        end
    end
    println("Verified repository head: ", head)
    notion === nothing && println("Notion verification skipped: NOTION_TOKEN not set.")
    failures
end

_default_manifest() = normpath(joinpath(@__DIR__, "..", "manifest.yaml"))
_default_state() = normpath(joinpath(@__DIR__, "..", "state.json"))
_default_page_map() = normpath(joinpath(@__DIR__, "..", "pages.local.yaml"))
_default_packet_dir() = normpath(joinpath(@__DIR__, "..", ".wiki-refresh", "packets"))

function _parse_options(args::Vector{String})
    options = Dict{String,Any}(
        "manifest" => _default_manifest(),
        "state" => _default_state(),
        "page_map" => _default_page_map(),
        "packet_dir" => _default_packet_dir(),
        "page" => nothing,
        "accept" => nothing,
        "bootstrap" => false,
    )
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--bootstrap"
            options["bootstrap"] = true; i += 1
        elseif arg in ("--manifest", "--state", "--page-map", "--packet-dir", "--page", "--accept")
            i == length(args) && throw(ArgumentError("$arg requires a value"))
            options[replace(arg[3:end], "-" => "_")] = args[i + 1]
            i += 2
        else
            throw(ArgumentError("unknown argument: $arg"))
        end
    end
    options
end

function _clients(manifest::Dict{String,Any})
    repo = manifest["repository"]
    token = get(ENV, "GITHUB_TOKEN", nothing)
    github = GitHubClient(String(repo["owner"]), String(repo["name"]), token)
    notion_token = get(ENV, "NOTION_TOKEN", nothing)
    notion = notion_token === nothing ? nothing :
        NotionClient(notion_token, String(manifest["notion"]["api_version"]))
    github, notion
end

function _command_check(manifest, state, github; only_page=nothing)
    state === nothing && begin
        println("No wiki/state.json exists; baseline review is required.")
        return EXIT_OWNER_REVIEW
    end
    head, drifts, broken = _collect_drifts(manifest, state, github; only_page)
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

function _command_verify(manifest, github, notion, page_map)
    failures = _verify_remote(manifest, github, notion, page_map)
    isempty(failures) && begin println("VERIFY: PASS"); return EXIT_CLEAN end
    println("\nVERIFY: FAIL")
    foreach(x -> println("  * ", x), failures)
    EXIT_BROKEN_SOURCE
end

function _accept_page!(manifest, state, github, notion, page_map,
        state_path::AbstractString, key::String)
    notion === nothing && throw(ArgumentError("NOTION_TOKEN is required for --accept"))
    haskey(page_map, key) || throw(ArgumentError("private Notion mapping missing for '$key'"))
    pages = manifest["pages"]
    haskey(pages, key) || throw(ArgumentError("unknown page key: $key"))
    page = pages[key]
    branch = String(manifest["repository"]["branch"])
    head, tree = _tree_blobs(github, branch)
    for rawpath in get(page, "sources", Any[])
        haskey(tree, String(rawpath)) ||
            throw(ArgumentError("cannot accept page with missing source: $rawpath"))
    end
    _update_verified_marker!(notion, page_map[key], head) ||
        throw(ArgumentError("page '$key' has no exact Verified against marker"))
    issue_cache = Dict{Int,Dict{String,Any}}()
    pr_cache = Dict{Int,Dict{String,Any}}()
    state["pages"][key] = _snapshot_page(page, head, github, issue_cache, pr_cache)
    state["repository"] = Dict("head" => head, "branch" => branch)
    state["updated_at"] = string(now(UTC))
    _json_write_file(state_path, state)
    println("Accepted reconciliation for ", key, " at ", head)
    EXIT_CLEAN
end

function _command_reconcile(manifest, state, github, notion, page_map, options)
    state_path = String(options["state"])
    if Bool(options["bootstrap"])
        state !== nothing && throw(ArgumentError("state already exists; refusing bootstrap"))
        _json_write_file(state_path, _bootstrap_state(manifest, github))
        println("Created initial wiki state at ", state_path)
        println("No GitHub or Notion writes were made.")
        return EXIT_CLEAN
    end
    accept = options["accept"]
    accept !== nothing && begin
        state === nothing && throw(ArgumentError("cannot --accept without state"))
        return _accept_page!(manifest, state, github, notion, page_map,
            state_path, String(accept))
    end
    state === nothing && return EXIT_OWNER_REVIEW
    only_page = options["page"]
    head, drifts, broken = _collect_drifts(manifest, state, github;
        only_page = only_page === nothing ? nothing : String(only_page))
    !isempty(broken) && return EXIT_BROKEN_SOURCE
    isempty(drifts) && begin println("NO_MATERIAL_WIKI_DRIFT"); return EXIT_CLEAN end
    semantic_count = 0
    for drift in drifts
        _print_drift(drift)
        if drift.mode == "semantic"
            semantic_count += 1
            packet = _write_packet(String(options["packet_dir"]), drift)
            println("  semantic packet: ", packet)
        else
            println("  mechanical page detected; v1 requires explicit --accept after review")
            semantic_count += 1
        end
    end
    println("\n", semantic_count, " page(s) require bounded review.")
    EXIT_OWNER_REVIEW
end

function main(args=ARGS)
    try
        argv = collect(String, args)
        command = isempty(argv) ? "check" : popfirst!(argv)
        options = _parse_options(argv)
        manifest = _load_manifest(String(options["manifest"]))
        _validate_manifest(manifest)
        state = _load_state(String(options["state"]))
        page_map = _load_page_map(String(options["page_map"]))
        github, notion = _clients(manifest)

        command == "check" && return _command_check(manifest, state, github;
            only_page = options["page"] === nothing ? nothing : String(options["page"]))
        command == "verify" && return _command_verify(manifest, github, notion, page_map)
        command == "reconcile" && return _command_reconcile(
            manifest, state, github, notion, page_map, options)
        println(stderr, "Unknown command: $command")
        return 2
    catch error
        println(stderr, "wiki-refresh error: ", sprint(showerror, error))
        error isa RemoteError && return EXIT_BROKEN_SOURCE
        error isa ArgumentError && return EXIT_OWNER_REVIEW
        rethrow()
    end
end

end
