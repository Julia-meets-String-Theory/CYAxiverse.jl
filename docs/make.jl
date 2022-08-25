push!(LOAD_PATH,"../src/")
using Documenter
using CYAxiverse

makedocs(
    sitename = "CYAxiverse.jl",
    authors = "Viraf M. Mehta",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://vmmhep.github.io/CYAxiverse.jl/stable/"),
    modules = [CYAxiverse],
    pages = [
        "Home" => "index.md"
        "User guide" => "userguide.md"
        "Examples" => "examples.md"
        "API" => "api.md"
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(branch = "gh-pages",
    repo = "github.com/vmmhep/CYAxiverse.jl.git",
    devbranch = "dev",
    target ="build",
    deps = nothing,
    make = nothing,
    push_preview =true)