using CYAxiverse
using HDF5
using LinearAlgebra
using Printf

"""
    compute_axion_data(h11::Int, np::Int, cy::Int, data_dir::String; threshold::Float64=0.5)

Computes axion spectra (masses, decay constants, quartic couplings) and vacua 
statistics/locations for a specific geometry specified by (h11, np, cy) in data_dir.
"""
function compute_axion_data(h11::Int, np::Int, cy::Int, data_dir::String; threshold::Float64=0.5)
    # Set the input search directory dynamically if provided
    if !isempty(data_dir) && isdir(data_dir)
        ENV["CYAXIVERSE_DATA_DIR"] = data_dir
    end

    # 1. Construct Geometry Index
    geom_idx = CYAxiverse.structs.GeometryIndex(h11, np, cy)
    
    # 2. Read potential & geometric data
    pot_data = CYAxiverse.read.potential(geom_idx)
    geom_data = CYAxiverse.read.geometry(geom_idx)
    
    # 3. Calculate Axion Spectrum (High-Precision Masses, Decay Constants, Quartic Couplings)
    spectrum = CYAxiverse.generate.hp_spectrum(pot_data.K, pot_data.L, pot_data.Q)
    
    # 4. Calculate Vacua Statistics & Locations
    vac_est = CYAxiverse.generate.vacua_estimate(geom_idx; threshold=threshold)
    vac_id = CYAxiverse.generate.vacua_id(pot_data.L, pot_data.Q; threshold=threshold)
    
    return Dict(
        "geom_idx" => geom_idx,
        "spectrum" => spectrum,
        "vacua_estimate" => vac_est,
        "vacua_locations" => vac_id,
        "potential" => pot_data,
        "geometry" => geom_data
    )
end

# Command-Line Interface Execution
if !isempty(ARGS)
    if length(ARGS) < 4
        println("Usage: julia run_axion_analysis.jl <h11> <np> <cy> <data_dir>")
        exit(1)
    end

    h11_in   = parse(Int, ARGS[1])
    np_in    = parse(Int, ARGS[2])
    cy_in    = parse(Int, ARGS[3])
    dir_in   = ARGS[4]

    println("==================================================")
    @printf("Processing Geometry: h11 = %d, np = %d, cy = %d\n", h11_in, np_in, cy_in)
    @printf("Data Directory: %s\n", dir_in)
    println("==================================================")

    results = compute_axion_data(h11_in, np_in, cy_in, dir_in)

    # --- Print Spectra Results ---
    println("\n[+] Axion Spectrum Summary:")
    println("  - Mass Eigenvalues (log10 eV):")
    println("    ", results["spectrum"]["m"])
    println("  - Decay Constants f_K (log10 M_Planck):")
    println("    ", results["spectrum"]["fK"])
    println("  - Perturbative Decay Constants f_pert:")
    println("    ", results["spectrum"]["fpert"])
    
    # --- Print Vacua Results ---
    println("\n[+] Vacua Statistics & Locations:")
    println("  - Estimated Total Vacua Count: ", results["vacua_estimate"].vac)
    println("  - Is Qhat Square Matrix: ", results["vacua_estimate"].issquare == 1 ? "Yes" : "No")
    
    if haskey(results["vacua_locations"], "θ̃∥")
        println("  - Minima Coordinates Matrix (θ̃∥):")
        display(results["vacua_locations"]["θ̃∥"])
    elseif haskey(results["vacua_locations"], "vac")
        println("  - Verified Vacua Count: ", results["vacua_locations"]["vac"])
    end
    println("==================================================")
end
# julia run_axion_analysis.jl 10 20 1 "./my_data_dir"