### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 24cd43c0-d41a-435a-8328-587ae4141bc1
begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "notebooks"))
    ENV["PYTHON"] = "/opt/homebrew/Caskroom/miniforge/base/envs/cytools/bin/python"
end

# ╔═╡ 3fcb377e-8cf6-11f1-be4d-5518d6c7b2b0
begin
	using CYAxiverse
	using PlutoUI
	using LinearAlgebra
	using Printf
end

# ╔═╡ 3fcb6b04-8cf6-11f1-979e-73e67f4aeb16
md"# Axion Spectra & Vacua Statistics Explorer
Configure the geometry parameters and directory structure below to calculate the axion masses, decay constants, and vacuum locations."

# ╔═╡ 3fcb6b5e-8cf6-11f1-853f-9b00196ab955
md"""
### Input Parameters & Directory Structure

- **h11**: $(@bind h11 NumberField(1:500, default=10))
- **np (polytope/triangulation index)**: $(@bind np NumberField(1:100000, default=20))
- **cy (Calabi-Yau index)**: $(@bind cy NumberField(1:100, default=1))
- **Data Directory**: $(@bind data_dir TextField(default="./data"))
- **Threshold**: $(@bind threshold NumberField(0.01:0.01:1.0, default=0.5))
"""

# ╔═╡ 3fcb6bba-8cf6-11f1-be26-d9ecccbd63c8
function compute_axion_notebook(h11::Int, np::Int, cy::Int, data_dir::String, threshold::Float64)
    if !isempty(data_dir) && isdir(data_dir)
        ENV["CYAXIVERSE_DATA_DIR"] = data_dir
    end

    geom_idx = CYAxiverse.structs.GeometryIndex(h11, np, cy)
    pot_data = CYAxiverse.read.potential(geom_idx)
    
    spectrum = CYAxiverse.generate.pq_spectrum(pot_data.K, pot_data.L, pot_data.Q)
    vac_est = CYAxiverse.generate.vacua_estimate(geom_idx; threshold=threshold)
    vac_locations = CYAxiverse.generate.vacua_id(pot_data.L, pot_data.Q; threshold=threshold)

    return (; geom=geom_idx, spec=spectrum, vac_est=vac_est, vac_loc=vac_locations)
end

# ╔═╡ 7e29b339-efbe-41ec-a78e-143f67a121a9
pot_data = CYAxiverse.read.potential(10,5,1)

# ╔═╡ 48ccc20a-51df-44f8-a5da-be6bd5dc2669
Matrix{Int}(pot_data.Q')

# ╔═╡ 3fcb6c1a-8cf6-11f1-99ce-c1fe0d56f614
res = compute_axion_notebook(h11, np, cy, data_dir, threshold)

# ╔═╡ ee05dce3-8af3-4b72-b9b6-90523db1cb9e
h11, np, cy

# ╔═╡ 3fcb6c30-8cf6-11f1-88c2-9f7bfdd9e5d6
md"""
## Summary Results for Geometry `(h11=$(h11), np=$(np), cy=$(cy))`

* **Total Vacua Estimate:** **$(res.vac_est.vac)**
* **Charge Basis Status:** $(res.vac_est.issquare == 1 ? "Square Charge Matrix Qhat" : "Non-square Charge Matrix (Extra Rows = $(res.vac_est.extrarows))")
"""

# ╔═╡ 3fcb6c76-8cf6-11f1-8b14-697291e2a278
md"""
### Axion Spectrum

* **Axion Mass Spectra ($\log_{10}$ eV):**
  `$(res.spec["m"])`

* **Kinetic Decay Constants $f_K$ ($\log_{10} M_{Planck}$):**
  `$(res.spec["fK"])`

* **Perturbative Decay Constants $f_{pert}$:**
  `$(res.spec["fpert"])`
"""

# ╔═╡ 3fcb6ca8-8cf6-11f1-a198-ef189b1162d9
md"""
### Vacuum Locations Matrix ($\tilde{\theta}_{min}$)
"""

# ╔═╡ 3fcb6cc6-8cf6-11f1-8397-73bb64709ec4
if haskey(res.vac_loc, "θ̃∥")
    res.vac_loc["θ̃∥"]
elseif haskey(res.vac_loc, "θ̃min")
    res.vac_loc["θ̃min"]
else
    "No explicit vacuum location matrix available for this geometry setup."
end

# ╔═╡ Cell order:
# ╠═24cd43c0-d41a-435a-8328-587ae4141bc1
# ╠═3fcb377e-8cf6-11f1-be4d-5518d6c7b2b0
# ╠═3fcb6b04-8cf6-11f1-979e-73e67f4aeb16
# ╠═3fcb6b5e-8cf6-11f1-853f-9b00196ab955
# ╠═3fcb6bba-8cf6-11f1-be26-d9ecccbd63c8
# ╠═7e29b339-efbe-41ec-a78e-143f67a121a9
# ╠═48ccc20a-51df-44f8-a5da-be6bd5dc2669
# ╠═3fcb6c1a-8cf6-11f1-99ce-c1fe0d56f614
# ╠═ee05dce3-8af3-4b72-b9b6-90523db1cb9e
# ╠═3fcb6c30-8cf6-11f1-88c2-9f7bfdd9e5d6
# ╠═3fcb6c76-8cf6-11f1-8b14-697291e2a278
# ╠═3fcb6ca8-8cf6-11f1-a198-ef189b1162d9
# ╠═3fcb6cc6-8cf6-11f1-8397-73bb64709ec4
