#!/usr/bin/env julia

"""
Model-stage driver for arXiv:2412.12012 Algorithm 1 (priority 4 of the
fuzzy-axions model-stage reproduction; see
`validation/fuzzy_axions_2412_12012_kaehler_qcd_model_count_scope_20260818.md`
Sec. 6, "Acceptance tests for the 3,348 comparison").

Reads the per-(X,FRST) Kähler-point export written by
`scripts/reproduce_fuzzy_axions_h11_4.py --model-stage` -- an HDF5 file with
one group `records/<i>` per accepted h21_plus_zero-class record (`Q`, `tau`,
`cy_volume`, `inverse_metric`, matching Priority 1's
`_export_kaehler_point`) -- calls
`CYAxiverse.paper_benchmarks.enumerate_fuzzy_axion_models` (Priority 3) once
per record under a single documented `(gs, W0)` convention shared across the
whole run, and writes per-record and total model counts plus full per-model
detail rows to an output HDF5 file for the Python driver to assemble into
the final JSON summary.

Usage:
    julia --project=. scripts/fuzzy_axion_model_stage_driver.jl \\
        <input.h5> <output.h5> <gs> <w0_real> <w0_imag>

Array-order gotcha (verified empirically before writing this script, not
assumed): HDF5.jl reads an N-D dataset written by h5py with its dimensions
*reversed* relative to the NumPy shape it was written with -- a Python
`(3, 4)` array round-trips as Julia `size (4, 3)`, and the values are the
literal transpose, not merely a relabelling (confirmed live: a 3x4 arange
array read back in Julia equals its own Python-side transpose element for
element). Every 2D dataset read here is therefore explicitly `permutedims`'d
back to the Python-documented shape before use. 1D datasets and scalars are
unaffected (also confirmed live) and are read as-is.
"""

using CYAxiverse
using HDF5

function main()
    length(ARGS) == 5 || error(
        "usage: julia fuzzy_axion_model_stage_driver.jl <input.h5> <output.h5> <gs> <w0_real> <w0_imag>")
    input_path, output_path, gs_arg, w0_real_arg, w0_imag_arg = ARGS
    gs = parse(Float64, gs_arg)
    w0 = Complex{Float64}(parse(Float64, w0_real_arg), parse(Float64, w0_imag_arg))

    prefactor_P = CYAxiverse.paper_benchmarks.fuzzy_axion_prefactor_P(gs)
    superpotential = CYAxiverse.paper_benchmarks.fuzzy_axion_flux_superpotential(w0)

    record_count = h5open(file -> read(file, "record_count"), input_path, "r")

    model_count_per_record = zeros(Int, record_count)
    model_record_index = Int[]
    model_axion_index = Int[]
    model_qcd_divisor_index = Int[]
    model_lambda = Float64[]
    model_mass_reference_log10_ev = Float64[]
    model_tau_reference = Float64[]

    h5open(input_path, "r") do file
        for i in 0:(record_count - 1)
            group = file["records/$i"]
            # Q, inverse_metric are 2D -- undo the h5py/HDF5.jl dimension
            # reversal (see module docstring) to recover the Python-documented
            # (h11, N) / (h11, h11) shapes enumerate_fuzzy_axion_models expects.
            Q = Matrix{Int}(permutedims(read(group, "Q")))
            tau = Vector{Float64}(read(group, "tau"))
            cy_volume = Float64(read(group, "cy_volume"))
            inverse_metric = Matrix{Float64}(permutedims(read(group, "inverse_metric")))

            kahler_pot = CYAxiverse.paper_benchmarks.fuzzy_axion_kahler_potential(cy_volume)
            gravitino_mass = CYAxiverse.paper_benchmarks.fuzzy_axion_gravitino_mass(
                prefactor_P, kahler_pot, superpotential; mplanck_ev=1.0)

            models = try
                CYAxiverse.paper_benchmarks.enumerate_fuzzy_axion_models(
                    Q, tau, cy_volume, prefactor_P, gravitino_mass, inverse_metric)
            catch exception
                @error "enumerate_fuzzy_axion_models failed" record_index=i Q tau cy_volume inverse_metric gravitino_mass
                rethrow(exception)
            end

            model_count_per_record[i + 1] = length(models)
            for model in models
                push!(model_record_index, i)
                push!(model_axion_index, model.axion_index)
                push!(model_qcd_divisor_index, model.qcd_divisor_index)
                push!(model_lambda, model.lambda)
                push!(model_mass_reference_log10_ev, model.mass_reference_log10_ev)
                push!(model_tau_reference, model.tau_reference)
            end
        end
    end

    total_model_count = sum(model_count_per_record)

    h5open(output_path, "w") do file
        write(file, "record_count", record_count)
        write(file, "total_model_count", total_model_count)
        write(file, "model_count_per_record", model_count_per_record)
        write(file, "model_record_index", model_record_index)
        write(file, "model_axion_index", model_axion_index)
        write(file, "model_qcd_divisor_index", model_qcd_divisor_index)
        write(file, "model_lambda", model_lambda)
        write(file, "model_mass_reference_log10_ev", model_mass_reference_log10_ev)
        write(file, "model_tau_reference", model_tau_reference)
        write(file, "gs", gs)
        write(file, "w0_real", real(w0))
        write(file, "w0_imag", imag(w0))
        write(file, "prefactor_P", prefactor_P)
    end

    println("wrote $(output_path): $(record_count) records, $(total_model_count) total models")
end

main()
