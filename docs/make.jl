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
        canonical = "https://julia-meets-string-theory.github.io/CYAxiverse.jl/stable/"),
    # Compatibility aliases point to already-documented benchmark modules.
    checkdocs = :exports,
    modules = [CYAxiverse],
    pages = [
        "Home" => "index.md",
        "User guide" => "userguide.md",
        "Local axion-photon scan" => "axion_photon.md",
        "Examples" => "examples.md",
        "API" => "api.md"
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
if get(ENV, "CI", "false") == "true"
    deploydocs(
        branch = "gh-pages",
        repo = "github.com/Julia-meets-String-Theory/CYAxiverse.jl.git",
        devbranch = "dev",
        target = "build",
        deps = nothing,
        make = nothing,
        push_preview = true,
    )
end
