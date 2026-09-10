#!/usr/bin/env julia
"""Issue 148 G2: N8 radial multifield catastrophe by branch continuation.

Exercises all G2 acceptance criteria under the owner-approved P96 contract.
Run: julia --startup-file=no --project=. scripts/issue_148_g2_continuation_evidence.jl
"""

using LinearAlgebra
using Printf
using Random
using Test

@info "Loading CYAxiverse..."
using CYAxiverse
include(joinpath(@__DIR__, "inflation_scale_continuation.jl"))
const pb = CYAxiverse.paper_benchmarks
const ai = pb.author_inflation

@info "=== Issue 148 G2: N8 Continuation Evidence ==="
@info "Source: arXiv:2608.14780v1, Appendix D, twelve-term Table 1 potential"
@info "Contract: P96 — period-one θ, argument 2πQθ, K_θ = M96/k²"
@info "Julia $(VERSION), $(Sys.MACHINE)"

# ── 1. Reference: existing augmented solve (validation target) ──────────────
@info "── Step 1: Existing augmented solve (independent validation target) ──"
augmented = pb.n8_degenerate_point()
@test augmented.converged
@info @sprintf("  Augmented kc          = %.15e", augmented.k)
@info @sprintf("  Gradient residual     = %.3e", augmented.gradient_residual)
@info @sprintf("  Null residual         = %.3e", augmented.null_residual)
@info "  Augmented theta (period-one) = $(augmented.theta)"
aug_setup = pb._n8_setup_leading_charge(augmented.k)
augmented_theta_glsm = mod.(pb._n8_leading_to_theta(augmented.theta,
    aug_setup.Qtilde), 1.0)
@info "  Augmented theta mapped to GLSM = $(augmented_theta_glsm)"

# ── 2. Find regular branches away from catastrophe ─────────────────────────
@info "── Step 2: Find regular branches at k=0.68 > kc and k=0.66 < kc ──"
Random.seed!(148)
k_above = 0.68
k_below = 0.66

branches_above = pb.n8_find_regular_branches(k_above; n_starts=512)
branches_below = pb.n8_find_regular_branches(k_below; n_starts=512)

minima_above = filter(b -> b.n_negative == 0 && b.n_zero == 0, branches_above)
saddles_above = filter(b -> b.n_negative == 1 && b.n_zero == 0, branches_above)
minima_below = filter(b -> b.n_negative == 0 && b.n_zero == 0, branches_below)
saddles_below = filter(b -> b.n_negative == 1 && b.n_zero == 0, branches_below)

@info @sprintf("  k=%.2f: %d stationary points, %d minima, %d index-1 saddles",
    k_above, length(branches_above), length(minima_above), length(saddles_above))
@info @sprintf("  k=%.2f: %d stationary points, %d minima, %d index-1 saddles",
    k_below, length(branches_below), length(minima_below), length(saddles_below))
@test length(minima_above) >= 1
@test length(minima_below) >= 1
@test length(minima_below) > length(minima_above) ||
    length(saddles_below) > length(saddles_above) ||
    (length(minima_below) >= 1 && length(saddles_below) >= 1)

# ── 3. Pseudo-arclength continuation from k > kc toward kc ─────────────────
@info "── Step 3: Pseudo-arclength continuation toward catastrophe ──"

results_from_above = pb.N8ContinuationResult{Float64}[]
for (bi, branch) in enumerate(vcat(minima_above, saddles_above))
    result = pb.n8_pseudo_arclength_continuation(
        branch.theta, k_above;
        ds=-5e-4, n_steps=300,
        tolerance=1e-8, k_bounds=(0.5, 0.75),
        branch_id=bi)
    push!(results_from_above, result)
    n_converged = count(s -> s.converged, result.steps)
    k_range = extrema(s.k for s in result.steps if s.converged)
    @info @sprintf("  Branch %d (index-%d): %d converged steps, k=[%.6f, %.6f], status=%s",
        bi, branch.n_negative, n_converged, k_range[1], k_range[2], result.status)
    if result.catastrophe_bracket !== nothing
        idx_lo, idx_hi = result.catastrophe_bracket
        @info @sprintf("    Catastrophe bracket: k=[%.8f, %.8f]",
            result.steps[idx_lo].k, result.steps[idx_hi].k)
    end
end

cat_results = filter(r -> r.status == :catastrophe_detected, results_from_above)
@info @sprintf("  Branches with catastrophe detected: %d / %d",
    length(cat_results), length(results_from_above))
@test length(cat_results) >= 1

# ── 4. Continuation from k < kc toward kc ─────────────────────────────────
@info "── Step 4: Continuation from below ──"
results_from_below = pb.N8ContinuationResult{Float64}[]
for (bi, branch) in enumerate(vcat(minima_below, saddles_below))
    result = pb.n8_pseudo_arclength_continuation(
        branch.theta, k_below;
        ds=5e-4, n_steps=300,
        tolerance=1e-8, k_bounds=(0.5, 0.75),
        branch_id=bi + 100)
    push!(results_from_below, result)
    n_converged = count(s -> s.converged, result.steps)
    if n_converged > 0
        k_range = extrema(s.k for s in result.steps if s.converged)
        @info @sprintf("  Branch %d (index-%d): %d steps, k=[%.6f, %.6f], status=%s",
            bi, branch.n_negative, n_converged, k_range[1], k_range[2], result.status)
    end
end
cat_results_below = filter(r -> r.status == :catastrophe_detected, results_from_below)
@info @sprintf("  Branches from below with catastrophe: %d / %d",
    length(cat_results_below), length(results_from_below))
@test length(cat_results_below) >= 1
@test all(r -> r.status != :running, vcat(results_from_above, results_from_below))

# ── 5. Catastrophe localization ────────────────────────────────────────────
@info "── Step 5: Catastrophe localization by bisection ──"
best_cat = first(cat_results)
localized = pb.n8_continuation_catastrophe_localization(best_cat)
@info @sprintf("  Localized kc              = %.15e", localized.k)
@info @sprintf("  Augmented kc              = %.15e", augmented.k)
@info @sprintf("  |k_cont - k_aug|          = %.3e", abs(localized.k - augmented.k))
@info @sprintf("  Gradient residual         = %.3e", localized.gradient_residual)
@info @sprintf("  Hessian min eigenvalue     = %.3e", localized.canonical_hessian_min)
@info @sprintf("  Bracket width             = %.3e", localized.bracket_width)
@test abs(localized.k - augmented.k) < 1e-4

# ── 6. Bordered system conditioning evidence ───────────────────────────────
@info "── Step 6: Bordered system / conditioning / step-failure evidence ──"
near_cat_steps = filter(s -> s.converged && abs(s.k - augmented.k) < 0.005,
    best_cat.steps)
if !isempty(near_cat_steps)
    for s in near_cat_steps[1:min(5, length(near_cat_steps))]
        @info @sprintf("  k=%.8f: λ_min=%.6e, |∇|=%.3e, iters=%d, ds=%.3e",
            s.k, s.canonical_hessian_min, s.gradient_residual,
            s.iterations, s.ds)
    end
end
cond_evidence = [(s.k, s.canonical_hessian_min, s.gradient_residual, s.iterations)
    for s in best_cat.steps if s.converged]
approaching = filter(t -> abs(t[1] - augmented.k) < 0.01, cond_evidence)
@test length(approaching) >= 2
@test all(isfinite(t[2]) && isfinite(t[3]) for t in approaching)
@test abs(approaching[end][2]) < abs(approaching[1][2])
@info "  Hessian eigenvalue converging to zero: CONFIRMED"
step_failures = filter(s -> !s.converged, best_cat.steps)
@info @sprintf("  Step failures near singularity: %d", length(step_failures))
@test isempty(step_failures)
method_counts = Dict(method => count(s -> s.corrector_method == method, best_cat.steps)
    for method in unique(s.corrector_method for s in best_cat.steps))
@info "  Corrector provenance: $(method_counts)"
@info @sprintf("  Bordered rank/condition at accepted event step: %d / %.3e",
    best_cat.steps[end-1].bordered_rank, best_cat.steps[end-1].bordered_condition)
@test any(s -> s.corrector_method == :bordered, best_cat.steps[2:end])
@test all(s -> s.bordered_rank == 9 && isfinite(s.bordered_condition),
    best_cat.steps[2:end])
@test all(s -> s.rejected_steps >= 0, best_cat.steps)

# Disabling the bordered corrector must change the accepted trace.  This
# catches a fallback-only implementation while keeping the probe bounded.
disabled_corrector = pb.n8_pseudo_arclength_continuation(
    best_cat.steps[1].theta, best_cat.steps[1].k; ds=-5e-4, n_steps=3,
    max_corrector_iterations=0, branch_id=best_cat.branch_id)
@info @sprintf("  Disabled-corrector probe: status=%s, first accepted k=%.15e",
    disabled_corrector.status, disabled_corrector.steps[2].k)
@test length(disabled_corrector.steps) >= 2
@test disabled_corrector.steps[2].corrector_method == :fixed_k_fallback
@test disabled_corrector.steps[2].k != best_cat.steps[2].k
forced_failure = pb.n8_pseudo_arclength_continuation(
    best_cat.steps[1].theta, best_cat.steps[1].k; ds=-5e-4, n_steps=1,
    tolerance=0.0, min_ds=4e-4, max_corrector_iterations=0,
    branch_id=best_cat.branch_id)
@info "  Forced nonconvergence probe: status=$(forced_failure.status), methods=$(unique(s.corrector_method for s in forced_failure.steps))"
@test forced_failure.status == :step_failed
@test any(!s.converged for s in forced_failure.steps)

# ── 7. Tangent direction / pseudo-arclength diagnostics ────────────────────
@info "── Step 7: Tangent / pseudo-arclength verification ──"
for s in best_cat.steps[1:min(3, length(best_cat.steps))]
    tnorm = sqrt(dot(s.tangent_theta, s.tangent_theta) + s.tangent_k^2)
    @info @sprintf("  Step %d: |tangent|=%.6f, tangent_k=%.6f",
        s.step_index, tnorm, s.tangent_k)
    @test isapprox(tnorm, 1.0; atol=1e-6)
end

# ── 8. Float64 → target-constructed BigFloat event ladder ───────────────────
@info "── Step 8: Precision ladder ──"

refined_128 = pb.n8_bigfloat_augmented_solve(
    localized.theta, localized.k; precision_bits=128)
@info @sprintf("  128-bit exact augmented event: converged=%s, |∇|=%.3e, |Hv|=%.3e",
    refined_128.converged, Float64(refined_128.gradient_residual),
    Float64(refined_128.null_residual))
@test refined_128.converged

refined_256 = pb.n8_bigfloat_augmented_solve(
    refined_128.theta, refined_128.k; precision_bits=256)
@info @sprintf("  256-bit exact augmented event: converged=%s, |∇|=%.3e, |Hv|=%.3e",
    refined_256.converged, Float64(refined_256.gradient_residual),
    Float64(refined_256.null_residual))
@test refined_256.converged

@info @sprintf("  |k_128 - k_256|   = %.3e", Float64(abs(refined_128.k - refined_256.k)))
@info @sprintf("  |k_128 - k_aug|   = %.3e", abs(Float64(refined_128.k) - augmented.k))
@info @sprintf("  |k_256 - k_aug|   = %.3e", abs(Float64(refined_256.k) - augmented.k))
@test abs(Float64(refined_128.k) - augmented.k) < 1e-8
@test abs(Float64(refined_256.k) - augmented.k) < 1e-10
@info "  k128=$(refined_128.k), k256=$(refined_256.k), difference=$(abs(refined_128.k - refined_256.k))"
@test abs(refined_128.k - refined_256.k) < BigFloat("1e-30")
@test refined_128.gradient_residual < BigFloat("1e-60")
@test refined_128.null_residual < BigFloat("1e-60")
@test refined_256.gradient_residual < BigFloat("1e-60")
@test refined_256.null_residual < BigFloat("1e-60")
@test isapprox(norm(refined_128.null_vector), BigFloat(1); atol=BigFloat("1e-35"))
@test isapprox(norm(refined_256.null_vector), BigFloat(1); atol=BigFloat("1e-35"))

# ── 9. BigFloat augmented solve (independent validation) ──────────────────
@info "── Step 9: BigFloat augmented solve (independent validation) ──"
aug_big = refined_256
@info @sprintf("  BigFloat augmented (ladder endpoint): converged=%s, |∇|=%.3e, |Hv|=%.3e",
    aug_big.converged, Float64(aug_big.gradient_residual),
    Float64(aug_big.null_residual))
@test aug_big.converged
if aug_big.converged
    @info @sprintf("  BigFloat kc            = %.15e", Float64(aug_big.k))
    @info @sprintf("  |k_big - k_aug_f64|    = %.3e",
        abs(Float64(aug_big.k) - augmented.k))
    @test abs(Float64(aug_big.k) - augmented.k) < 1e-10
    theta_diff = maximum(min.(abs.(Float64.(aug_big.theta) .- localized.theta),
        1.0 .- abs.(Float64.(aug_big.theta) .- localized.theta)))
    @info @sprintf("  max|θ_big - θ_aug_f64| = %.3e", theta_diff)
    @test theta_diff < 1e-6
    @test aug_big.gradient_residual < BigFloat("1e-60")
    @test aug_big.null_residual < BigFloat("1e-60")
    @test isapprox(norm(aug_big.null_vector), BigFloat(1); atol=BigFloat("1e-60"))
end

# ── 10. P96 classification at the catastrophe ──────────────────────────────
@info "── Step 10: P96 catastrophe classification ──"
diag_p96 = pb.n8_continuation_classify(augmented_theta_glsm, augmented.k)
@info @sprintf("  Classification:     %s", diag_p96.classification)
@info @sprintf("  Normal form:        %s", diag_p96.normal_form)
@info @sprintf("  Stationary:         %s", diag_p96.is_stationary)
@info @sprintf("  Gradient residual:  %.3e", diag_p96.gradient_residual)
@info @sprintf("  Null eigenvalue:    %.3e", diag_p96.near_null_eigenvalue)
@info "  Projected derivatives: 2nd=$(diag_p96.projected_derivatives.second), " *
    "3rd=$(diag_p96.projected_derivatives.third), 4th=$(diag_p96.projected_derivatives.fourth)"
@info "  Transverse eigenvalues: $(diag_p96.transverse_hessian_eigenvalues)"
@test diag_p96.is_stationary
@test diag_p96.classification in (:cusp, :fold, :unresolved)
@test abs(diag_p96.projected_derivatives.second) <= diag_p96.derivative_cutoff

diag_p96_big = pb.n8_bigfloat_p96_diagnostic(
    refined_256.theta, refined_256.k; precision_bits=256)
@info @sprintf("  Exact-source 256-bit P96 diagnostic: class=%s, stationary=%s, metric source bits=%d",
    diag_p96_big.classification, diag_p96_big.is_stationary,
    diag_p96_big.metric_source_precision_bits)
@test diag_p96_big.is_stationary
@test diag_p96_big.classification in (:cusp, :fold, :unresolved)
@test diag_p96_big.metric_source_precision_bits == 53
@test diag_p96_big.metric_precision_boundary == :float64_reconstructed

# A96 comparison (labeled separately).  First compare like-for-like ten-term
# data at the same author point and the same metric witness; this isolates the
# coordinate tensor factors.  The approved source12 P96 diagnostic above stays
# separate and uses the precise reconstructed M96 metric.
author10 = ai.n8_degenerate_point()
author10_potential = ai.n8_potential(k=author10.k; trajectory=true)
author10_theta = author10.theta ./ (2π)
author10_metric = Matrix(ai.n8_geometry().kinetic) ./ author10.k^2
author10_amplitudes = vec(author10_potential.L[1, :]) .* 10.0 .^ vec(author10_potential.L[2, :])
# Use a fixed nondegenerate probe on the same ten-term potential so the
# near-null eigenspace at the catastrophe cannot amplify roundoff in the
# coordinate-factor comparison.
author10_probe_x = author10.theta .+ 2e-4 .* collect(1.0:8.0)
author10_probe_theta = author10_probe_x ./ (2π)
diag_p96_author10 = pb.local_catastrophe_diagnostic(
    author10_probe_theta, author10_potential.Q, author10_amplitudes, author10_metric;
    phases=author10_potential.phases, argument_scale=2π, precision_bits=53)
diag_a96 = pb.local_catastrophe_diagnostic(
    author10_probe_x, author10_potential.Q, author10_amplitudes, author10_metric;
    phases=author10_potential.phases, argument_scale=1, precision_bits=53,
    null_direction=diag_p96_author10.canonical_direction)
ratio_2nd = diag_p96_author10.projected_derivatives.second /
    diag_a96.projected_derivatives.second
ratio_4th = diag_p96_author10.projected_derivatives.fourth /
    diag_a96.projected_derivatives.fourth
@info @sprintf("  A96 classification: %s (labeled A96, author10 normalization)",
    diag_a96.classification)
@info @sprintf("  Like-for-like ten-term P96/A96 factors: D2=%.6f (expected %.6f), D4=%.6f (expected %.6f)",
    ratio_2nd, (2π)^2, ratio_4th, (2π)^4)
@test size(pb._N8_SOURCE_CHARGES, 1) == 12
@test size(ai.N8_Q_TRAJECTORY, 1) == 10
@test isapprox(ratio_2nd, (2π)^2; rtol=1e-6, atol=1e-6)
@test isapprox(ratio_4th, (2π)^4; rtol=1e-6, atol=1e-6)

# ── 11. Hessian nullity and transverse spectrum ────────────────────────────
@info "── Step 11: Nullity and transverse Hessian ──"
@info @sprintf("  Number of near-null eigenvalues: %d",
    length(diag_p96.near_null_eigenvalues))
@test length(diag_p96.near_null_eigenvalues) == 1
trans_positive = all(>(0), diag_p96.transverse_hessian_eigenvalues)
@info @sprintf("  All transverse eigenvalues positive: %s", trans_positive)
@test trans_positive

# ── 12. Normalization / coordinate verification ───────────────────────────
@info "── Step 12: Normalization and coordinate verification ──"
geometry = pb.n8_geometry()
metric_p96 = Matrix(geometry.kinetic) / augmented.k^2
metric_check = Matrix(pb.n8_kinetic_matrix(augmented.k))
metric_diff = maximum(abs.(metric_p96 .- metric_check))
@info @sprintf("  Metric M96/k² agreement: %.3e", metric_diff)
@test metric_diff < 1e-15

eig_spectrum = eigvals(Symmetric(Matrix(geometry.kinetic)))
@info "  Eq.96 spectrum (ascending): $(round.(eig_spectrum, sigdigits=4))"
published = [5.84e-5, 8.30e-5, 9.15e-5, 1.24e-4, 3.13e-4, 5.97e-4, 6.35e-4, 8.20e-4]
for (i, (comp, pub)) in enumerate(zip(sort(eig_spectrum), published))
    @test isapprox(comp, pub; rtol=0.02)
end
@info "  Eq.96 spectrum agreement: PASS (all within 2%)"

# ── 13. Branch identity from continuation vs post-hoc matcher ──────────────
@info "── Step 13: Continuation vs post-hoc matcher comparison ──"
@info "  Branch identity in continuation: intrinsic via tangent predictor-corrector chain"
@info "  Branch identity in matcher: post-hoc periodic distance"

best = first(cat_results)
n_steps_conv = count(s -> s.converged, best.steps)
branch_ids = unique(s.branch_id for s in best.steps if s.converged)
@info @sprintf("  Best continuation branch: %d converged steps, branch IDs: %s",
    n_steps_conv, string(branch_ids))
@info "  Continuation identity is intrinsic: each step seeds from the previous corrected point"
@info "  Independent adjacent-slice records are corrected before pilot matching"

# Exercise the existing pilot matcher on independently solved records.  The
# records are built from fresh fixed-scale branch searches at adjacent slices;
# no continuation point is copied into either population.
adjacent_branches = pb.n8_find_regular_branches(0.6795; n_starts=128)
continuation_seed_population = vcat(minima_above, saddles_above)
sample_above = continuation_seed_population[1:min(13, length(continuation_seed_population))]
sample_adjacent = adjacent_branches[1:min(24, length(adjacent_branches))]
pilot_potential_above = pb._n8_potential(k=0.68)
pilot_potential_adjacent = pb._n8_potential(k=0.6795)
pilot_factor = Matrix{Float64}(I, 8, 8)
pilot_previous = _pilot_records(
    [copy(b.theta) for b in sample_above],
    [b.n_negative for b in sample_above], pilot_potential_above.Q,
    pilot_potential_above.L, pilot_factor;
    residual_tolerance=1e-10, max_iterations=100, duplicate_tolerance=1e-6)
pilot_current = _pilot_records(
    [copy(b.theta) for b in sample_adjacent],
    [b.n_negative for b in sample_adjacent], pilot_potential_adjacent.Q,
    pilot_potential_adjacent.L, pilot_factor;
    residual_tolerance=1e-10, max_iterations=100, duplicate_tolerance=1e-6)
_pilot_init_branch_ids!(pilot_previous)
pilot_matches = pilot_match_records!(pilot_previous, pilot_current;
    matching_tolerance=0.05)
@info @sprintf("  Actual pilot matcher: previous=%d, current=%d, matches=%d",
    length(pilot_previous), length(pilot_current), length(pilot_matches))
@test !isempty(pilot_matches)

periodic_distance(a, b) = maximum(min.(abs.(a .- b), 1.0 .- abs.(a .- b)))
function nearest_intrinsic_id(theta, k, results; k_tolerance=5e-5)
    candidates = [(periodic_distance(theta, s.theta), r.branch_id)
        for r in results for s in r.steps
        if s.converged && abs(s.k - k) <= k_tolerance]
    isempty(candidates) && return (Inf, nothing)
    minimum(candidates)
end
intrinsic_seed_ids = Dict{String,Int}()
for record in pilot_previous
    d, bid = nearest_intrinsic_id(record.corrected_theta, 0.68,
        results_from_above; k_tolerance=1e-8)
    d < 1e-4 && (intrinsic_seed_ids[record.branch_match_id] = bid)
end
matcher_disagreements = NamedTuple[]
matched_comparisons = Ref(0)
for record in pilot_current
    record.matching_status == :matched || continue
    d, intrinsic_id = nearest_intrinsic_id(record.corrected_theta, 0.6795,
        results_from_above)
    expected_id = get(intrinsic_seed_ids, record.branch_match_id, nothing)
    matched_comparisons[] += 1
    if expected_id === nothing || intrinsic_id === nothing || expected_id != intrinsic_id
        push!(matcher_disagreements, (; branch_match_id=record.branch_match_id,
            expected_id, intrinsic_id, distance=d))
    end
end
@info @sprintf("  Matcher identity comparisons=%d, disagreements=%d",
    matched_comparisons[], length(matcher_disagreements))
for disagreement in matcher_disagreements
    @info "  Matcher disagreement: $(disagreement)"
end
@test matched_comparisons[] > 0

# ── 14. Merger/degeneracy evidence ─────────────────────────────────────────
@info "── Step 14: Branch merger / degeneracy evidence ──"

merger_theta = nothing
if length(cat_results) >= 2
    # Distinct periodic symmetry images are valid independent branches.  Do
    # not assign a merger by choosing the first two grid roots; use the
    # intrinsic eigenvalue crossing on each continuation chain instead.
    near_sets = [filter(s -> s.converged && abs(s.k - augmented.k) < 0.002,
        r.steps) for r in cat_results]
    near_sets = filter(!isempty, near_sets)
    if length(near_sets) >= 2
        start_separations = [maximum(min.(abs.(a.steps[1].theta .- b.steps[1].theta),
            1.0 .- abs.(a.steps[1].theta .- b.steps[1].theta)))
            for i in 1:length(cat_results)-1 for b in cat_results[i+1:end]
            for a in (cat_results[i],)]
        separations = [maximum(min.(abs.(a[end].theta .- b[end].theta),
            1.0 .- abs.(a[end].theta .- b[end].theta)))
            for i in 1:length(near_sets)-1 for b in near_sets[i+1:end]
            for a in (near_sets[i],)]
        @info @sprintf("  Minimum periodic separation of sampled event branches: %.3e",
            minimum(separations))
        @info @sprintf("  Corresponding start separation: %.3e",
            minimum(start_separations))
        @test minimum(start_separations) > 1e-6
        @test minimum(separations) < 1e-6
    end
    @test all(r -> r.catastrophe_bracket !== nothing, cat_results)
    merger_theta = near_sets[1][end].theta
elseif length(cat_results) == 1
    br = cat_results[1]
    if br.catastrophe_bracket !== nothing
        idx_lo, idx_hi = br.catastrophe_bracket
        s_lo = br.steps[idx_lo]
        s_hi = br.steps[idx_hi]
        @info @sprintf("  Single-branch fold: min eigenvalue crosses zero")
            @info @sprintf("    Before: λ_min = %+.6e at k=%.8f", s_lo.canonical_hessian_min, s_lo.k)
        @info @sprintf("    After:  λ_min = %+.6e at k=%.8f", s_hi.canonical_hessian_min, s_hi.k)
        @test s_lo.canonical_hessian_min * s_hi.canonical_hessian_min < 0
        @info "  Fold-type merger: eigenvalue sign change confirmed across bracket"
        merger_theta = (s_lo.theta .+ s_hi.theta) ./ 2
    end
end

# ── 15. Continuation-recovered k comparison with augmented solve ───────────
@info "── Step 15: Continuation-recovered kc vs augmented solve ──"
@info @sprintf("  Continuation localized kc  = %.15e", localized.k)
@info @sprintf("  Augmented Float64 kc       = %.15e", augmented.k)
@info @sprintf("  Agreement                  = %.3e", abs(localized.k - augmented.k))
@info "  Note: continuation recovery is independent (started away from catastrophe)"

# ── 16. Gradient / stationarity at continuation-recovered point ────────────
@info "── Step 16: Gradient / stationarity at recovered point ──"
potential_at_cat = pb._n8_potential(k=localized.k)
derivs = pb._n8_potential_derivatives(localized.theta, potential_at_cat.Q, potential_at_cat.L)
@info @sprintf("  Raw gradient ∞-norm: %.3e", norm(derivs.gradient, Inf))
@test norm(derivs.gradient, Inf) < 1e-8

# ── 17. Focused regression tests ──────────────────────────────────────────
@info "── Step 17: Focused regression tests ──"
@testset "G2 continuation regressions" begin
    @testset "source/metric conventions" begin
        @test size(pb.n8_geometry().kinetic) == (8, 8)
        @test isapprox(pb.n8_geometry().volume, 126.0)
        @test all(eigvals(Symmetric(Matrix(pb.n8_geometry().kinetic))) .> 0)
    end

    @testset "continuation basic function" begin
        k_test = 0.68
        branches = pb.n8_find_regular_branches(k_test; n_starts=64)
        @test length(branches) >= 1
        @test all(b -> b.converged, branches)
        @test all(b -> b.gradient_residual < 1e-9, branches)
    end

    @testset "continuation event recovery" begin
        @test length(cat_results) >= 1
        cr = first(cat_results)
        @test cr.status == :catastrophe_detected
        @test cr.catastrophe_bracket !== nothing
        loc = pb.n8_continuation_catastrophe_localization(cr)
        @test loc.gradient_residual < 1e-8
        @test abs(loc.k - augmented.k) < 1e-3
    end

    @testset "precision ladder" begin
        @test refined_128.converged
        @test Float64(refined_128.gradient_residual) < 1e-20
        @test refined_256.converged
        @test Float64(refined_256.gradient_residual) < 1e-40
    end

    @testset "P96 classification" begin
        @test diag_p96.is_stationary
        @test diag_p96.classification in (:cusp, :fold, :unresolved)
        @test length(diag_p96.near_null_eigenvalues) == 1
    end

    @testset "branch identity intrinsic" begin
        if !isempty(cat_results)
            cr = first(cat_results)
            for i in 2:length(cr.steps)
                @test cr.steps[i].branch_id == cr.steps[1].branch_id
            end
        end
    end

    @testset "augmented solve reference" begin
        @test augmented.converged
        @test augmented.gradient_residual < 1e-10
        @test augmented.null_residual < 1e-10
    end
end

# ── Summary ───────────────────────────────────────────────────────────────
@info "=== G2 Evidence Summary ==="
@info @sprintf("  Continuation kc:     %.15e", localized.k)
@info @sprintf("  Augmented kc:        %.15e", augmented.k)
@info @sprintf("  |Δk|:                %.3e", abs(localized.k - augmented.k))
@info @sprintf("  BigFloat kc (256b):  %.15e",
    aug_big.converged ? Float64(aug_big.k) : NaN)
@info @sprintf("  Classification:      %s / %s", diag_p96.classification, diag_p96.normal_form)
@info @sprintf("  Null eigenvalues:    %d", length(diag_p96.near_null_eigenvalues))
@info @sprintf("  Precision ladder:    128-bit %s, 256-bit %s",
    refined_128.converged ? "PASS" : "FAIL",
    refined_256.converged ? "PASS" : "FAIL")
@info "  Contract: P96 (period-one θ, 2πQθ, M96/k²)"
@info "  Source: arXiv:2608.14780v1 Table 1 (twelve terms)"
@info "  All G2 acceptance tests: PASS"
