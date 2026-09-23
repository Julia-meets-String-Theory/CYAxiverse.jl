"""Build and deploy release-neutral CYAxiverse documentation.

The source tree is shared by development and release documentation. Routing is
selected from a ref plus immutable release/publication manifest evidence. A
canonical tag without verified manifest evidence fails closed.
"""

const _CANONICAL_PUBLIC_TAG = r"^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$"
const _CANONICAL_VERSION = r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$"
const _FULL_SHA = r"^[0-9a-f]{40}$"

struct DocsRouteError <: Exception
    message::String
end

Base.showerror(io::IO, err::DocsRouteError) = print(io, err.message)

function _bounded_version(value::AbstractString, pattern::Regex)
    identity = match(pattern, String(value))
    identity === nothing && return false
    all(component -> tryparse(UInt32, component) !== nothing, identity.captures)
end

_canonical_tag(value::AbstractString) = _bounded_version(value, _CANONICAL_PUBLIC_TAG)
_canonical_final(value::AbstractString) = _bounded_version(value, _CANONICAL_VERSION)

"""Return the documentation channel for one verified deployment context."""
function docs_route(; ref::AbstractString,
                      manifest_status::AbstractString = "",
                      release_line::AbstractString = "",
                      release_version::AbstractString = "",
                      release_sha::AbstractString = "",
                      principal_main_sha::AbstractString = "",
                      stable_requested::Bool = false)
    ref_string = String(ref)
    status = lowercase(String(manifest_status))

    if isempty(ref_string) || startswith(ref_string, "refs/pull/") ||
       (status == "preview" && !startswith(ref_string, "refs/tags/"))
        return (channel = :preview, ref = ref_string, tag = nothing,
                version = nothing, line = nothing, stable = false)
    end

    if ref_string == "refs/heads/vmm"
        return (channel = :development, ref = ref_string, tag = nothing,
                version = nothing, line = :principal, stable = false)
    end

    tag_prefix = "refs/tags/"
    tag = startswith(ref_string, tag_prefix) ? ref_string[length(tag_prefix) + 1:end] : ""
    if ref_string != tag_prefix * tag || !_canonical_tag(tag)
        throw(DocsRouteError("DOCS_ROUTE_INVALID: ref is not vmm or a canonical public tag"))
    end
    status == "verified" || throw(DocsRouteError(
        "DOCS_ROUTE_UNVERIFIED: canonical tag requires verified release manifest evidence"))
    valid_maintenance_line = occursin(r"^maintenance/(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$", release_line)
    release_line == "principal" || valid_maintenance_line || throw(DocsRouteError(
        "DOCS_ROUTE_INVALID: verified tag requires principal or maintenance release line"))
    match(_FULL_SHA, release_sha) !== nothing || throw(DocsRouteError(
        "DOCS_ROUTE_INVALID: verified tag requires certified release commit"))

    version = tag[2:end]
    _canonical_final(release_version) && release_version == version || throw(DocsRouteError(
        "DOCS_ROUTE_INVALID: tag and release-manifest version disagree"))

    stable = false
    channel = :versioned
    if stable_requested
        release_line == "principal" || throw(DocsRouteError(
            "DOCS_STABLE_INVALID: maintenance releases cannot advance stable"))
        match(_FULL_SHA, principal_main_sha) !== nothing || throw(DocsRouteError(
            "DOCS_STABLE_INVALID: stable requires current principal main identity"))
        release_sha == principal_main_sha || throw(DocsRouteError(
            "DOCS_STABLE_INVALID: certified release commit is not current principal main"))
        stable = true
        channel = :stable
    end

    return (channel = channel, ref = ref_string, tag = tag, version = version,
            line = Symbol(release_line), stable = stable)
end

function _env_bool(name::AbstractString)
    lowercase(strip(get(ENV, name, "false"))) in ("1", "true", "yes")
end

function docs_route_from_environment()
    docs_route(
        ref = get(ENV, "CYAX_DOCS_REF", get(ENV, "GITHUB_REF", "")),
        manifest_status = get(ENV, "CYAX_DOCS_MANIFEST_STATUS", ""),
        release_line = get(ENV, "CYAX_DOCS_RELEASE_LINE", ""),
        release_version = get(ENV, "CYAX_DOCS_RELEASE_VERSION", ""),
        release_sha = get(ENV, "CYAX_DOCS_RELEASE_SHA", ""),
        principal_main_sha = get(ENV, "CYAX_DOCS_PRINCIPAL_MAIN_SHA", ""),
        stable_requested = _env_bool("CYAX_DOCS_STABLE"),
    )
end

function _deploy_versions(route)
    route.channel === :preview && return Union{String,Pair{String,String}}[]
    versions = Union{String,Pair{String,String}}[]
    stable_tag = get(ENV, "CYAX_DOCS_STABLE_TAG", "")
    # Documenter's `versions` pairs control selector links and symlinks.
    # Pin stable to the verifier's current principal tag. This preserves
    # stable while a maintenance tag gets its own immutable folder; the
    # built-in `v^` selector would incorrectly choose that patch.
    if !isempty(stable_tag)
        _canonical_tag(stable_tag) || throw(DocsRouteError(
            "DOCS_STABLE_CONTEXT_INVALID: stable principal tag is not canonical"))
        push!(versions, "stable" => stable_tag)
    elseif route.channel !== :development
        throw(DocsRouteError(
            "DOCS_STABLE_CONTEXT_MISSING: verified principal tag is required"))
    end
    # Keep one persistent selector for every canonical vX.Y.Z tag and dev.
    # The patch selector is a string entry; pairs are literal selector labels
    # and target directories. `devbranch` below selects the source branch that
    # populates Documenter's `dev` deployment directory.
    push!(versions, "v#.#.#")
    push!(versions, "dev" => "dev")
    versions
end

function _prepare_documenter_context!(route)
    get(ENV, "GITHUB_EVENT_NAME", "") == "workflow_dispatch" || return nothing
    route.channel === :versioned || throw(DocsRouteError(
        "DOCS_DISPATCH_INVALID: workflow_dispatch requires a verified versioned tag"))
    get(ENV, "CYAX_DOCS_MANIFEST_STATUS", "") == "verified" || throw(DocsRouteError(
        "DOCS_DISPATCH_UNVERIFIED: workflow_dispatch requires verified manifest evidence"))
    verified_ref = "refs/tags/$(route.tag)"
    get(ENV, "CYAX_DOCS_REF", "") == verified_ref || throw(DocsRouteError(
        "DOCS_DISPATCH_REF_MISMATCH: verified documentation ref is not exact"))
    get(ENV, "CYAX_DOCS_TAG_REF", "") == verified_ref || throw(DocsRouteError(
        "DOCS_DISPATCH_REF_MISMATCH: verified tag ref is not exact"))

    # Documenter 1.19 selects release vs dev from GITHUB_REF.  On manual
    # publication dispatch, use only the tag already validated against the
    # immutable publication/released manifests above.
    ENV["GITHUB_REF"] = verified_ref
    nothing
end

function _documenter_subfolder(route, documenter)
    _prepare_documenter_context!(route)
    decision = documenter.deploy_folder(
        documenter.GitHubActions();
        branch = "gh-pages",
        repo = "github.com/Julia-meets-String-Theory/CYAxiverse.jl.git",
        devbranch = "vmm",
        devurl = "dev",
        push_preview = true,
    )
    decision.all_ok || throw(DocsRouteError(
        "DOCS_DEPLOY_CONTEXT_INVALID: Documenter rejected the deployment context"))
    documenter.determine_deploy_subfolder(decision, _deploy_versions(route))
end

# Keep routing tests independent of the documentation dependencies.
route_only = get(ENV, "CYAX_DOCS_ROUTE_ONLY", "")
if route_only == "true" || route_only == "versions" || route_only == "documenter"
    route = docs_route_from_environment()
    if route_only == "versions"
        println(join((item isa Pair ? string(first(item), ":", last(item)) : item
                      for item in _deploy_versions(route)), ","))
    elseif route_only == "documenter"
        @eval using Documenter
        println(Base.invokelatest(_documenter_subfolder, route, getfield(Main, :Documenter)))
    else
        println(route.channel)
    end
    exit(0)
end

route = docs_route_from_environment()
docs_channel_path = route.channel === :development ? "dev" :
                    route.channel === :stable ? "stable" :
                    route.channel === :versioned ? route.tag : "preview"
push!(LOAD_PATH, "../src/")
using Documenter
using CYAxiverse

makedocs(
    sitename = "CYAxiverse.jl",
    authors = "Viraf M. Mehta",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        size_threshold = 300 * 1024,
        size_threshold_warn = 300 * 1024,
        canonical = "https://julia-meets-string-theory.github.io/CYAxiverse.jl/$(docs_channel_path)/"),
    # Compatibility aliases point to already-documented benchmark modules.
    checkdocs = :exports,
    modules = [CYAxiverse],
    pages = [
        "Home" => "index.md",
        "Installation" => "installation.md",
        "User guide" => "userguide.md",
        "Pipelines" => "pipelines.md",
        "Local axion-photon scan" => "axion_photon.md",
        "Examples" => "examples.md",
        "API" => "api.md"
    ]
)

if _env_bool("DOCS_DEPLOY")
    route.channel === :preview && throw(DocsRouteError(
        "DOCS_ROUTE_INVALID: preview builds cannot deploy documentation"))
    _prepare_documenter_context!(route)
    deploydocs(
        branch = "gh-pages",
        repo = "github.com/Julia-meets-String-Theory/CYAxiverse.jl.git",
        devbranch = "vmm",
        versions = _deploy_versions(route),
        target = "build",
        deps = nothing,
        make = nothing,
        push_preview = true,
    )
end
