push!(LOAD_PATH,"../src/")
using Documenter
using CYAxiverse

makedocs(
    sitename = "CYAxiverse.jl",
    authors = "Viraf M. Mehta",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        size_threshold = 300 * 1024,
        size_threshold_warn = 300 * 1024,
        canonical = "https://julia-meets-string-theory.github.io/CYAxiverse.jl/dev/"),
    # Compatibility aliases point to already-documented benchmark modules.
    checkdocs = :exports,
    modules = [CYAxiverse],
    pages = [
        "Home" => "index.md",
        "User guide" => "userguide.md",
        "Pipelines" => "pipelines.md",
        "Local axion-photon scan" => "axion_photon.md",
        "Examples" => "examples.md",
        "API" => "api.md"
    ]
)

if get(ENV, "DOCS_DEPLOY", "false") == "true"
    deploydocs(
        branch = "gh-pages",
        repo = "github.com/Julia-meets-String-Theory/CYAxiverse.jl.git",
        devbranch = "vmm",
        target = "build",
        deps = nothing,
        make = nothing,
        push_preview = true,
    )
end
