using ArbNumerics
using CYAxiverse
using HDF5
using LinearAlgebra
using Printf

function geometry_files(data_root::AbstractString, h11::Int)
    directory = joinpath(data_root, "data", "h11_" * lpad(h11, 3, '0'))
    files = String[]
    for (root, _, names) in walkdir(directory), name in names
        name == "cyax.h5" && push!(files, joinpath(root, name))
    end
    sort!(files)
end

function load_potential(path::AbstractString)
    h5open(path, "r") do file
        Q = Int.(HDF5.read(file, "cytools/potential/Q"))
        L = HDF5.read(file, "cytools/potential/L")
        K = Hermitian(inv(HDF5.read(file, "cytools/geometric/Kinv")))
        return K, L, Q
    end
end

function benchmark_geometry(path::AbstractString, h11::Int, index::Int)
    K, L, Q = load_potential(path)
    threshold = Float64(log10(CYAxiverse.generate.constants()["Hubble"]))

    GC.gc()
    selector_seconds = @elapsed selected = CYAxiverse.generate.LQtilde(Q, L)

    GC.gc()
    inertia_150_seconds = @elapsed count_150 = CYAxiverse.generate.physical_mode_inertia_count(
        K, selected.Ltilde, selected.Qtilde, threshold, 150)
    GC.gc()
    inertia_200_seconds = @elapsed count_200 = CYAxiverse.generate.physical_mode_inertia_count(
        K, selected.Ltilde, selected.Qtilde, threshold, 200)

    label = "h11=$(h11), sample=$(index)"
    GC.gc()
    hybrid_seconds = @elapsed spectrum = CYAxiverse.generate.pq_hybrid_physical_spectrum(
        K, L, Q; prec=200, quartics=false, label)

    minimum_mass, maximum_mass = isempty(spectrum.m) ? (NaN, NaN) : extrema(spectrum.m)
    @printf(
        "%d\t%d\t%d\t%.6f\t%d\t%.6f\t%d\t%.6f\t%.6f\t%.12g\t%.12g\n",
        h11, index, size(Q, 2), selector_seconds,
        count_150, inertia_150_seconds, count_200, inertia_200_seconds,
        hybrid_seconds, minimum_mass, maximum_mass,
    )
    flush(stdout)
end

function main(args)
    length(args) >= 2 || error("usage: benchmark_hybrid_scaling.jl DATA_ROOT H11 [LIMIT]")
    data_root = abspath(args[1])
    h11 = parse(Int, args[2])
    limit = length(args) >= 3 ? parse(Int, args[3]) : typemax(Int)
    files = geometry_files(data_root, h11)
    resize!(files, min(length(files), limit))

    println("h11\tsample\tinstantons\tselector_s\tcount_150\tinertia_150_s\tcount_200\tinertia_200_s\thybrid_200_s\tminimum_mass\tmaximum_mass")
    for (index, path) in enumerate(files)
        benchmark_geometry(path, h11, index)
    end
end

main(ARGS)
