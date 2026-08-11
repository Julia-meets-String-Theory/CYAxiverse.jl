"""Benchmark the full reference spectrum against a narrow mass window."""

using LinearAlgebra
using CYAxiverse

"""Build a diagonal sparse-charge reference with a compact log-scale span."""
function spectrum_benchmark_input(h11::Int)
    h11 > 1 || throw(ArgumentError("h11 must be greater than one"))
    Q = zeros(Int, h11, h11 + 1)
    Q[:, 1:h11] .= Matrix{Int}(I, h11, h11)
    L = zeros(Float64, 2, h11 + 1)
    L[1, :] .= 1.0
    L[2, 1:h11] .= collect(range(-20.0, stop=-29.9, length=h11))
    Hermitian(Matrix{Float64}(I, h11, h11)), L, Q
end

"""Measure one deterministic full-spectrum/window pair."""
function benchmark_spectrum_window(; h11::Int=100, prec::Int=100,
        window_modes::Int=5)
    K, L, Q = spectrum_benchmark_input(h11)
    reference = CYAxiverse.generate.pq_physical_spectrum(
        K, L, Q; threshold_log10=-Inf, prec)
    first_mode = max(1, (h11 - window_modes) ÷ 2)
    last_mode = first_mode + window_modes - 1
    first_mode <= last_mode <= length(reference.m) ||
        throw(ArgumentError("window_modes must fit inside the reference spectrum"))
    min_log10_mass = reference.m[first_mode]
    max_log10_mass = reference.m[last_mode]

    # Warm both paths so the report measures the numerical kernels rather than
    # package loading and first-call compilation.
    CYAxiverse.generate.pq_physical_spectrum(
        K, L, Q; threshold_log10=-Inf, prec)
    CYAxiverse.generate.pq_window_spectrum(
        K, L, Q; min_log10_mass, max_log10_mass, prec,
        confirm=false, quartics=false)

    GC.gc()
    reference_seconds = @elapsed CYAxiverse.generate.pq_physical_spectrum(
        K, L, Q; threshold_log10=-Inf, prec)
    GC.gc()
    window = CYAxiverse.generate.pq_window_spectrum(
        K, L, Q; min_log10_mass, max_log10_mass, prec,
        confirm=false, quartics=false)
    GC.gc()
    window_seconds = @elapsed CYAxiverse.generate.pq_window_spectrum(
        K, L, Q; min_log10_mass, max_log10_mass, prec,
        confirm=false, quartics=false)
    GC.gc()
    reference_allocated = @allocated CYAxiverse.generate.pq_physical_spectrum(
        K, L, Q; threshold_log10=-Inf, prec)
    GC.gc()
    window_allocated = @allocated CYAxiverse.generate.pq_window_spectrum(
        K, L, Q; min_log10_mass, max_log10_mass, prec,
        confirm=false, quartics=false)

    window.mode_indices == collect(first_mode - 1:last_mode - 1) ||
        throw(AssertionError("window mode indices do not match the reference"))
    window.diagnostics.certified ||
        throw(AssertionError("benchmark window did not certify"))
    !window.diagnostics.fallback_used ||
        throw(AssertionError("benchmark window unexpectedly fell back"))

    (; h11, prec, window_modes, min_log10_mass, max_log10_mass,
        reference_seconds, window_seconds,
        speedup=reference_seconds / window_seconds,
        reference_allocated, window_allocated,
        fallback_used=window.diagnostics.fallback_used)
end

if abspath(PROGRAM_FILE) == @__FILE__
    println(benchmark_spectrum_window())
end
