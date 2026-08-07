#!/usr/bin/env julia

"""Reproduce the N=8 minima-count checkpoint from the 2023 paper."""

using CYAxiverse
using Printf

requested_starts = isempty(ARGS) ? 2048 : parse(Int, ARGS[1])
starts = max(requested_starts, 2048)
results = CYAxiverse.axion_benchmarks.n8_minima_scan(; starts=starts)

println("N=8 2023-paper minima checkpoint")
@printf("starts per scale: %d\n", starts)
for result in results
    @printf("k=%.8f  critical=%d  minima=%d\n",
            result.k, result.critical_count, result.minima_count)
end

expected = (5, 1)
observed = Tuple(result.minima_count for result in results)
if observed != expected
    error("unexpected N=8 minima counts: observed=$(observed), expected=$(expected)")
end
println("checkpoint passed")
