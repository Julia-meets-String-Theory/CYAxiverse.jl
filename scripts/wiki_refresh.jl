#!/usr/bin/env julia

using Pkg

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const WIKI_PROJECT = joinpath(REPO_ROOT, "wiki")

Pkg.activate(WIKI_PROJECT)

using CYAxiverseWikiRefresh

exit(CYAxiverseWikiRefresh.main(ARGS))
