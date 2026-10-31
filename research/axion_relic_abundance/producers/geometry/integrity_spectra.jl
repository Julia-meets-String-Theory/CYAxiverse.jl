# Phase 11: recompute the physical spectrum of sampled geometries independently from the MATERIALIZED files
# (fresh process, same entry point and settings as batch_physical_spectrum.jl) and compare with the batch outputs
# written into the spectra/ copy. Usage:
#   CYAXIVERSE_DATA_DIR=<materialized> julia --project=<code> integrity_spectra.jl FILES_TXT SPECTRA_ROOT OUT_TSV
using CYAxiverse, HDF5
files = filter(!isempty, readlines(ARGS[1])); spectra_root = ARGS[2]
thr = Float64(log10(CYAxiverse.generate.constants()["Hubble"]))
open(ARGS[3], "w") do io
    println(io, "file\trecomputed_modes\tbatch_modes\tmasses_identical\tfpert_identical\tmax_abs_dlog10m")
    for f in files
        m = match(r"h11_(\d+)/np_(\d+)/cy_(\d+)/cyax.h5", f); h, p, c = parse.(Int, m.captures)
        s = CYAxiverse.generate.pq_hybrid_physical_spectrum(CYAxiverse.structs.GeometryIndex(h, p, c);
                threshold_log10=thr, prec=500, quartics=true, mixed_quartics=true)
        mb, fb = h5open(joinpath(spectra_root, f), "r") do fh
            read(fh["spectrum/physical/m"]), read(fh["spectrum/physical/fpert_log10"])
        end
        fr = s.m .- 9 .- 0.5 .* s.λself
        same = length(s.m) == length(mb) && s.m == mb
        d = length(s.m) == length(mb) && !isempty(mb) ? maximum(abs.(s.m .- mb)) : (length(s.m) == length(mb) ? 0.0 : Inf)
        println(io, join((f, length(s.m), length(mb), same, length(fr) == length(fb) && fr == fb, d), "\t"))
    end
end
println("done")
