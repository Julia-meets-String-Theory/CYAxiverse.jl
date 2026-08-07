"""
    CYAxiverse.paper_benchmarks

Deterministic benchmark potentials and inflation trajectories used by the
vacua pipeline's reproduction targets. Charge matrices returned here follow
the package convention: axions are rows and instantons are columns.
"""
module paper_benchmarks

using LinearAlgebra
using NLsolve
using Optim
using ..generate: LQtilde, reduced_critical_points

include(joinpath(@__DIR__, "paper_benchmarks", "reduced_models.jl"))
include(joinpath(@__DIR__, "paper_benchmarks", "poly102_inflation.jl"))
include(joinpath(@__DIR__, "paper_benchmarks", "compatibility.jl"))

end
