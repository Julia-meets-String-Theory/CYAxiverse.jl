### A Pluto.jl notebook ###
# v0.19.0

using Markdown
using InteractiveUtils

# ╔═╡ cell-imports
using CYAxiverse
using PlutoUI
using LinearAlgebra
using Printf

# ╔═╡ cell-title
md"# Axion Spectra & Vacua Statistics Explorer
Configure the geometry parameters and directory structure below to calculate the axion masses, decay constants, and vacuum locations."

# ╔═╡ cell-inputs
md"""
### Input Parameters & Directory Structure

- **h11**: $(@bind h11 NumberField(1:500, default=10))
- **np (polytope/triangulation index)**: $(@bind np NumberField(1:100000, default=20))
- **cy (Calabi-Yau index)**: $(@bind cy NumberField(1:100, default=1))
- **Data Directory**: $(@bind data_dir TextField(default="./data"))
- **Threshold**: $(@bind threshold NumberField(0.01:0.01:1.0, default=0.5))
"""

# ╔═╡ cell-computation-function
function compute_axion_notebook(h11::Int, np::Int, cy::Int, data_dir::String, threshold::Float64)
    if !isempty(data_dir) && isdir(data_dir)
        ENV["CYAXIVERSE_DATA_DIR"] = data_dir
    end

    geom_idx = CYAxiverse.structs.GeometryIndex(h11, np, cy)
    pot_data = CYAxiverse.read.potential(geom_idx)
    
    spectrum = CYAxiverse.generate.hp_spectrum(pot_data.K, pot_data.L, pot_data.Q)
    vac_est = CYAxiverse.generate.vacua_estimate(geom_idx; threshold=threshold)
    vac_locations = CYAxiverse.generate.vacua_id(pot_data.L, pot_data.Q; threshold=threshold)

    return (geom=geom_idx, spec=spectrum, vac_est=vac_est, vac_loc=vac_locations)
end

# ╔═╡ cell-run
res = compute_axion_notebook(h11, np, cy, data_dir, threshold)

# ╔═╡ cell-output-summary
md"""
## Summary Results for Geometry `(h11=$(h11), np=$(np), cy=$(cy))`

* **Total Vacua Estimate:** **$(res.vac_est.vac)**
* **Charge Basis Status:** $(res.vac_est.issquare == 1 ? "Square Charge Matrix Qhat" : "Non-square Charge Matrix (Extra Rows = $(res.vac_est.extrarows))")
"""

# ╔═╡ cell-output-spectra
md"""
### Axion Spectrum

* **Axion Mass Spectra ($\log_{10}$ eV):**
  `$(res.spec["m"])`

* **Kinetic Decay Constants $f_K$ ($\log_{10} M_{Planck}$):**
  `$(res.spec["fK"])`

* **Perturbative Decay Constants $f_{pert}$:**
  `$(res.spec["fpert"])`
"""

# ╔═╡ cell-output-locations
md"""
### Vacuum Locations Matrix ($\tilde{\theta}_{min}$)
"""

# ╔═╡ cell-display-locations
if haskey(res.vac_loc, "θ̃∥")
    res.vac_loc["θ̃∥"]
elseif haskey(res.vac_loc, "θ̃min")
    res.vac_loc["θ̃min"]
else
    "No explicit vacuum location matrix available for this geometry setup."
end

# ╔═╡ Cell order:
# ╠═ cell-title
# ╠═ cell-imports
# ╠═ cell-inputs
# ╠═ cell-computation-function
# ╠═ cell-run
# ╠═ cell-output-summary
# ╠═ cell-output-spectra
# ╠═ cell-output-locations
# ╠═ cell-display-locations