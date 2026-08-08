### A Pluto.jl notebook ###
# v0.20.0
using Markdown
using InteractiveUtils

# ╔═╡ 8ef9d8b7-f1eb-4b0d-8c7a-4dd40391d7de
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "notebooks"))
    ENV["PYTHON"] = "/opt/homebrew/Caskroom/miniforge/base/envs/cytools/bin/python"
    using LinearAlgebra
    using HDF5
    using CYAxiverse
    import CYAxiverse.cytools_wrapper as cw
    import CYAxiverse.read as read_mod
    import CYAxiverse.minimizer as minimizer_mod
end

# ╔═╡ f84b40b0-1c9d-48f2-8a0c-55a8ea3a5dcd
md"This notebook generates a geometry from CYTools through the wrapper module, loads it back through the package read path, computes the potential data, and runs the minimizer."

# ╔═╡ 329d4ef6-7d30-4472-8d1f-8c41ef3f9fa7
begin
    h11 = 4
    poly_test = cw.fetch_polytopes(h11, 4, lattice="N", as_list=true, favorable=true)
    @assert !isempty(poly_test)
    cy = poly_test[1].triangulate().get_cy()
    cw.geometries(h11, cy, 1, 1)
    geom = read_mod.geometry(h11, 1, 1)
    pot = read_mod.potential(h11, 1, 1)
    (geom=geom, pot=pot)
end

# ╔═╡ 0401f7d3-0266-4df1-9d1d-5f75a2e80cb9
begin
    L_vals = read_mod.L_arb(h11, 1, 1)
    Q_mat = read_mod.Q(h11, 1, 1)
    x0 = zeros(size(Q_mat, 2))
    result = minimizer_mod.minimize(L_vals, Q_mat, x0)
    result
end

# ╔═╡ 6b736b91-8a73-4f12-8692-a2df-0f8c83565826
begin
    @assert size(geom.glsm_charges, 1) == h11
    @assert size(pot.L, 1) > 0
    @assert haskey(result, "logV")
    "End-to-end workflow completed: geometry loaded, potential computed, and minimizer returned a result."
end

# ╔═╡ Cell metadata
# ╠═╡ ""
