#!/usr/bin/env julia

"""Reproducible accuracy sweep for `classify_point`'s whitening change.

PR #87 replaced `analyze_inflation_candidates.jl::classify_point`'s two
explicit inverses (`inv(factor')` for the Hessian rotation, `inv(K)` for the
gradient norm) with triangular solves against a caller-owned `Cholesky`
factor. The commit message reported measuring both forms against a 512-bit
`BigFloat` reference over h11 = 10, 20, 40 and cond(K) = 1e6...1e14, and
found the two forms comparable -- both scaling like `eps * cond(K)`, neither
systematically better -- so the change shipped on cost and consistency
grounds, not accuracy. That sweep was never checked in, so the claim was not
independently reproducible from the repository. This script reproduces it.

For each (h11, cond(K)) cell it draws `--seeds` independent random points
(K's eigenbasis rotation and theta both vary per draw), computes the
canonical Hessian eigenvalues and gradient norm three ways -- the pre-#87
explicit-inverse form, the current whitened form, and a `BigFloat` reference
built from the same Q/L/theta/K -- and records each form's error against the
reference. Read-only: no HDF5 geometry is touched, and `Q`/`L`/`K` are
synthetic fixtures (`pseudo_Q`/`pseudo_L` plus a random Kähler metric at a
requested condition number), matching the fixture already pinned in
`test/runtests.jl`'s "whitened point classification" testset.

Output is a stdout summary table; curated results belong in `validation/`,
so this script does not write into the repository itself. Pass
`--output-csv PATH` to also dump one row per draw for offline analysis.
"""

using CYAxiverse
using LinearAlgebra
using Printf
using Random
using Statistics

const AnalyzeInflationCandidates = Module(:AnalyzeInflationCandidatesForSweep)
Base.include(AnalyzeInflationCandidates,
    joinpath(@__DIR__, "analyze_inflation_candidates.jl"))

# `classify_point` exactly as it stood before PR #87, kept here so the form
# being compared against the current implementation is pinned rather than
# reconstructed from memory. Mirrors `classify_point_explicit_inverse` in
# `test/runtests.jl`.
function classify_point_explicit_inverse(theta, Q, L, K)
    d = AnalyzeInflationCandidates.derivatives(theta, Q, L)
    factor = cholesky(Hermitian(K)).L
    to_theta = inv(factor')
    canonical_hessian = to_theta' * d.hessian * to_theta
    eigs = eigvals(Hermitian(canonical_hessian))
    gradnorm = sqrt(max(dot(d.gradient, inv(K) * d.gradient), 0.0))
    (; gradnorm, hessian_eigenvalues=eigs)
end

function classify_point_whitened(theta, Q, L, Kfactor::Cholesky)
    c = AnalyzeInflationCandidates.classify_point(theta, Q, L, Kfactor)
    (; gradnorm=c.gradnorm, hessian_eigenvalues=c.hessian_eigenvalues)
end

# Same formula as `classify_point`'s Hessian/gradient-norm computation,
# evaluated at high precision from the same Q/L/theta/K so this is a
# precision comparison, not an independent derivation.
function bigfloat_reference(theta, Q, L, K; precision_bits)
    setprecision(BigFloat, precision_bits) do
        Qbig = BigFloat.(Q)
        Lbig = BigFloat.(L)
        thetabig = BigFloat.(theta)
        Kbig = BigFloat.(K)
        logscale = Lbig[2, :]
        amplitudes = Lbig[1, :] .* BigFloat(10) .^ (logscale .- maximum(logscale))
        args = 2 * BigFloat(pi) .* (transpose(Qbig) * thetabig)
        gradient = 2 * BigFloat(pi) .* Qbig * (amplitudes .* sin.(args))
        weights = amplitudes .* cos.(args)
        hessian = 4 * BigFloat(pi)^2 .* Qbig * Diagonal(weights) * transpose(Qbig)
        Lfac = cholesky(Hermitian(Kbig)).L
        canonical_hessian = Lfac \ hessian / Lfac'
        eigs = eigvals(Hermitian(canonical_hessian))
        gradnorm = norm(Lfac \ gradient)
        (; gradnorm=Float64(gradnorm), hessian_eigenvalues=Float64.(eigs))
    end
end

# Same fixture shape as `classification_fixture` in `test/runtests.jl`'s
# "whitened point classification" testset, with `theta` also randomised per
# draw instead of fixed, since this sweep wants independent samples rather
# than a single pinned equivalence point.
function classification_fixture(h11, condition_number, rng)
    Q = Matrix(CYAxiverse.generate.pseudo_Q(h11, 1)')
    L = Matrix(CYAxiverse.generate.pseudo_L(h11, 1)')
    rotation = qr(randn(rng, h11, h11)).Q
    spectrum = exp10.(range(0, -log10(condition_number); length=h11))
    K = Matrix(Hermitian(rotation * Diagonal(spectrum) * rotation'))
    theta = 0.5 .+ 0.3 .* (2 .* rand(rng, h11) .- 1)
    Q, L, K, theta
end

scale_of(reference_eigs) = max(maximum(abs, reference_eigs), 1.0)

function hessian_rel_err(form_eigs, reference_eigs)
    maximum(abs.(Float64.(form_eigs) .- reference_eigs)) / scale_of(reference_eigs)
end

gradnorm_rel_err(form_gradnorm, reference_gradnorm) =
    abs(form_gradnorm - reference_gradnorm) / max(abs(reference_gradnorm), 1.0)

function run_cell(h11, cond, seeds, precision_bits, base_seed, csv_io)
    old_hess = Float64[]; new_hess = Float64[]
    old_grad = Float64[]; new_grad = Float64[]
    new_hess_wins = 0; new_grad_wins = 0
    for draw in 1:seeds
        rng = MersenneTwister(hash((base_seed, h11, cond, draw)))
        Q, L, K, theta = classification_fixture(h11, cond, rng)
        Kfactor = cholesky(Hermitian(K))
        old = classify_point_explicit_inverse(theta, Q, L, K)
        new = classify_point_whitened(theta, Q, L, Kfactor)
        reference = bigfloat_reference(theta, Q, L, K; precision_bits)

        oh = hessian_rel_err(old.hessian_eigenvalues, reference.hessian_eigenvalues)
        nh = hessian_rel_err(new.hessian_eigenvalues, reference.hessian_eigenvalues)
        og = gradnorm_rel_err(old.gradnorm, reference.gradnorm)
        ng = gradnorm_rel_err(new.gradnorm, reference.gradnorm)

        push!(old_hess, oh); push!(new_hess, nh)
        push!(old_grad, og); push!(new_grad, ng)
        nh < oh && (new_hess_wins += 1)
        ng < og && (new_grad_wins += 1)

        csv_io === nothing || @printf(csv_io, "%d,%.6e,%d,%.6e,%.6e,%.6e,%.6e\n",
            h11, cond, draw, oh, nh, og, ng)
    end
    (; h11, cond, seeds,
       old_hess_max=maximum(old_hess), new_hess_max=maximum(new_hess),
       old_hess_median=median(old_hess), new_hess_median=median(new_hess),
       old_grad_max=maximum(old_grad), new_grad_max=maximum(new_grad),
       old_grad_median=median(old_grad), new_grad_median=median(new_grad),
       new_hess_win_rate=new_hess_wins / seeds, new_grad_win_rate=new_grad_wins / seeds)
end

function _usage()
    println("""
    Usage: julia --project=. scripts/validate_classify_point_whitening_accuracy.jl [options]

      --h11 LIST            Comma-separated h11 values. Default: 10,20,40.
      --cond LIST           Comma-separated cond(K) values. Default: 1e6,1e8,1e10,1e12,1e14.
      --seeds N             Random draws per (h11,cond) cell. Default: 20.
      --precision-bits N    BigFloat reference precision. Default: 512.
      --seed-base N         Base seed folded into each draw's RNG. Default: 20260818.
      --output-csv PATH     Optional per-draw CSV (not written by default).
    """)
end

function _parse_args(args)
    options = (h11=[10, 20, 40], cond=[1e6, 1e8, 1e10, 1e12, 1e14],
        seeds=20, precision_bits=512, seed_base=20260818, output_csv="")
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("--help", "-h")
            _usage(); exit(0)
        elseif arg == "--h11"
            options = merge(options, (; h11=parse.(Int, split(args[i + 1], ","))))
            i += 1
        elseif arg == "--cond"
            options = merge(options, (; cond=parse.(Float64, split(args[i + 1], ","))))
            i += 1
        elseif arg == "--seeds"
            options = merge(options, (; seeds=parse(Int, args[i + 1])))
            i += 1
        elseif arg == "--precision-bits"
            options = merge(options, (; precision_bits=parse(Int, args[i + 1])))
            i += 1
        elseif arg == "--seed-base"
            options = merge(options, (; seed_base=parse(Int, args[i + 1])))
            i += 1
        elseif arg == "--output-csv"
            options = merge(options, (; output_csv=args[i + 1]))
            i += 1
        else
            error("unknown option: $arg")
        end
        i += 1
    end
    options.seeds > 0 || error("seeds must be positive")
    options.precision_bits >= 128 || error("precision-bits must be at least 128")
    options
end

function main(args=ARGS)
    options = _parse_args(args)
    csv_io = isempty(options.output_csv) ? nothing : open(options.output_csv, "w")
    csv_io === nothing || println(csv_io,
        "h11,cond,draw,old_hess_rel_err,new_hess_rel_err,old_grad_rel_err,new_grad_rel_err")

    println("classify_point whitening accuracy sweep")
    println("julia=$(VERSION)  precision_bits=$(options.precision_bits)  seeds/cell=$(options.seeds)")
    println()
    @printf("%-5s %-9s %14s %14s %10s   %14s %14s %10s\n",
        "h11", "cond(K)", "old max err", "new max err", "new<old%",
        "old grad max", "new grad max", "new<old%")

    cells = [run_cell(h11, cond, options.seeds, options.precision_bits, options.seed_base, csv_io)
             for h11 in options.h11, cond in options.cond]

    for cell in vec(cells)
        @printf("%-5d %-9.0e %14.3e %14.3e %9.0f%%   %14.3e %14.3e %9.0f%%\n",
            cell.h11, cell.cond, cell.old_hess_max, cell.new_hess_max,
            100 * cell.new_hess_win_rate, cell.old_grad_max, cell.new_grad_max,
            100 * cell.new_grad_win_rate)
    end

    total_draws = sum(cell.seeds for cell in vec(cells))
    hess_wins = sum(round(Int, cell.new_hess_win_rate * cell.seeds) for cell in vec(cells))
    grad_wins = sum(round(Int, cell.new_grad_win_rate * cell.seeds) for cell in vec(cells))
    println()
    @printf("whitened form closer on Hessian eigenvalues in %d/%d draws (%.1f%%)\n",
        hess_wins, total_draws, 100 * hess_wins / total_draws)
    @printf("whitened form closer on gradient norm in %d/%d draws (%.1f%%)\n",
        grad_wins, total_draws, 100 * grad_wins / total_draws)

    csv_io === nothing || close(csv_io)
    true
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
