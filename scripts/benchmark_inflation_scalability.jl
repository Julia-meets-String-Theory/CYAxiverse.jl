#!/usr/bin/env julia

"""Measure bounded, warm-process inflation trajectory pilots.

This is a trajectory microbenchmark for the fixed poly-102 fixture. It is not
an O(10^5)-geometry scan: the current reproduction path has no geometry
loader, worker pool, checkpoint store, or shard writer.

Usage: `julia --project=. scripts/benchmark_inflation_scalability.jl [COUNT [MAX_TIME]]`
"""

using CYAxiverse
using Printf

const Poly102 = CYAxiverse.axion_benchmarks.poly102_inflation
const DELTA_K = 1.5320548620798324e-3
const PRECISION_BITS = 64
const RELTOL = 1e-8
const ABSTOL = 1e-10
const MAX_STEP = 1.0
const INITIAL_STEP = 1e-5

pilot_count = isempty(ARGS) ? 1 : parse(Int, ARGS[1])
max_time = length(ARGS) < 2 ? 10.0 : parse(Float64, ARGS[2])
pilot_count > 0 || error("COUNT must be positive")
max_time > 0 || error("MAX_TIME must be positive")

trajectory(; delta_k=DELTA_K) = Poly102.n8_physical_gradient_flow(delta_k;
    precision_bits=PRECISION_BITS, max_time, max_step=MAX_STEP,
    initial_step=INITIAL_STEP, sample_count=1, reltol=RELTOL,
    abstol=ABSTOL, maxiters=1_000_000)

function main(pilot_count, max_time)
    # Warm compilation and method caches before measuring. The warmup is
    # reported separately and is deliberately excluded from the pilot rows.
    warmup = trajectory()
    GC.gc()
    baseline_rss = Sys.maxrss()

    println("# fixed poly-102 trajectory pilot; warmup excluded")
    @printf("# count=%d max_time=%.6g precision_bits=%d reltol=%.3g abstol=%.3g max_step=%.6g initial_step=%.6g\n",
        pilot_count, max_time, PRECISION_BITS, RELTOL, ABSTOL, MAX_STEP, INITIAL_STEP)
    println("run\twall_seconds\tallocated_bytes\toutput_bytes\tentered\tend_event\tterminated\tefolds\taccepted_steps")

    failures = 0
    total_wall = 0.0
    total_allocated = 0
    total_output = 0
    for run in 1:pilot_count
        GC.gc()
        measured = try
            @timed trajectory()
        catch error
            failures += 1
            println("$run\tFAIL\t$((sprint(showerror, error)))")
            continue
        end
        result = measured.value
        total_wall += measured.time
        total_allocated += measured.bytes
        output_bytes = Base.summarysize(result)
        total_output += output_bytes
        @printf("%d\t%.6f\t%d\t%d\t%s\t%s\t%s\t%.12g\t%d\n",
            run, measured.time, measured.bytes, output_bytes,
            result.entered_slow_roll, result.end_event, result.terminated,
            Float64(result.efolds), result.solver.accepted_steps)
    end

    completed = pilot_count - failures
    println("# completed=$completed failures=$failures peak_rss_bytes=$(Sys.maxrss()) baseline_rss_bytes=$baseline_rss")
    if completed > 0
        @printf("# mean_wall_seconds=%.6f mean_allocated_bytes=%.0f mean_output_bytes=%.0f\n",
            total_wall / completed, total_allocated / completed,
            total_output / completed)
    end
end

main(pilot_count, max_time)
