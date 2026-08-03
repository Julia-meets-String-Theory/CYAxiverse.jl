using CYAxiverse
using HDF5
using LinearAlgebra
using Printf

"""
    compute_axion_data(h11::Int, np::Int, cy::Int, data_dir::String; threshold::Float64=0.5)

Computes axion spectra (masses, decay constants, quartic couplings) and vacua 
statistics/locations for a specific geometry specified by (h11, np, cy) in data_dir.
"""
function save_axion_data(geom_idx, spectrum, vac_est, vac_id; threshold::Float64)
    h5open(CYAxiverse.filestructure.cyax_file(geom_idx), "r+") do file
        if haskey(file, "spectrum")
            HDF5.delete_object(file, "spectrum")
        end
        spectrum_group = create_group(file, "spectrum")
        masses_group = create_group(spectrum_group, "masses")
        masses_group["log10", deflate=9] = spectrum.m
        masses_group["sign", deflate=9] = spectrum.msign
        decay_group = create_group(spectrum_group, "decay")
        decay_group["fK", deflate=9] = spectrum.fK
        # Preserve the established HDF5 name. For a PQ spectrum this dataset
        # contains AxionSpectrum.f, the sequential PQ decay quantity.
        decay_group["fpert", deflate=9] = spectrum.f
        quartdiag_group = create_group(spectrum_group, "quartdiag")
        quartdiag_group["log10", deflate=9] = spectrum.λself
        quartdiag_group["sign", deflate=9] = spectrum.λselfsign
        quart31_group = create_group(spectrum_group, "quart31")
        quart31_group["index", deflate=9] = spectrum.λ31_i
        quart31_group["log10", deflate=9] = spectrum.λ31
        quart31_group["sign", deflate=9] = spectrum.λ31sign
        quart22_group = create_group(spectrum_group, "quart22")
        quart22_group["index", deflate=9] = spectrum.λ22_i
        quart22_group["log10", deflate=9] = spectrum.λ22
        quart22_group["sign", deflate=9] = spectrum.λ22sign

        if haskey(file, "vacua_pipeline")
            HDF5.delete_object(file, "vacua_pipeline")
        end
        vacua_group = create_group(file, "vacua_pipeline")
        vacua_group["threshold", deflate=9] = threshold
        vacua_group["estimate", deflate=9] = vac_est.vac
        vacua_group["issquare", deflate=9] = vac_est.issquare
        if hasproperty(vac_est, :extrarows)
            vacua_group["extrarows", deflate=9] = vac_est.extrarows
        end
        if haskey(vac_id, "vac")
            vacua_group["verified", deflate=9] = vac_id["vac"]
        end
        for (key, path) in (("θ̃min", "theta_min"), ("θ̃∥", "theta_parallel"))
            if haskey(vac_id, key)
                coordinates = vac_id[key]
                coordinates_group = create_group(vacua_group, path)
                coordinates_group["numerator", deflate=9] = Int.(numerator.(coordinates))
                coordinates_group["denominator", deflate=9] = Int.(denominator.(coordinates))
            end
        end
    end
end

function compute_axion_data(h11::Int, np::Int, cy::Int, data_dir::String; threshold::Float64=0.5, save::Bool=true)
    # Set the input search directory dynamically if provided
    if !isempty(data_dir) && isdir(data_dir)
        ENV["CYAXIVERSE_DATA_DIR"] = data_dir
    end

    # 1. Construct Geometry Index
    geom_idx = CYAxiverse.structs.GeometryIndex(h11, np, cy)
    
    # 2. Read potential & geometric data
    pot_data = CYAxiverse.read.potential(geom_idx)
    geom_data = CYAxiverse.read.geometry(geom_idx)
    
    # 3. Calculate the PQ mass-basis spectrum and quartic couplings.
    spectrum = CYAxiverse.generate.pq_spectrum(geom_idx)
    
    # 4. Calculate Vacua Statistics & Locations
    vac_est = CYAxiverse.generate.vacua_estimate(geom_idx; threshold=threshold)
    vac_id = CYAxiverse.generate.vacua_id(pot_data.L, pot_data.Q; threshold=threshold)
    if save
        save_axion_data(geom_idx, spectrum, vac_est, vac_id; threshold=threshold)
    end
    
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
if abspath(PROGRAM_FILE) == @__FILE__
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
    println("    ", results["spectrum"].m)
    println("  - Decay Constants f_K (log10 M_Planck):")
    println("    ", results["spectrum"].fK)
    println("  - Sequential PQ Decay Quantities f (stored as fpert):")
    println("    ", results["spectrum"].f)
    
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
