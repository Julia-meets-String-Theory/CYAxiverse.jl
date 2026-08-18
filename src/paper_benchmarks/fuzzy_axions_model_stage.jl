"""
Model-stage evaluator for arXiv:2412.12012 Algorithm 1 (Sec. 4.1, "Fuzzy
Axions and Associated Relics").

Phase 3 of the priority-2/3 model-stage scoping in
`validation/fuzzy_axions_2412_12012_kaehler_qcd_model_count_scope_20260818.md`.
Consumes: the Python export at the canonical-tip reference point t0
(`scripts/reproduce_fuzzy_axions_h11_4.py`'s `--export-kaehler-points`: `Q`,
`prime_divisor_volumes`, `cy_volume`, `inverse_metric`); Phase 1's scalar
formulas (`fuzzy_axion_prefactor_P`, `fuzzy_axion_kahler_potential`,
`fuzzy_axion_gravitino_mass`); and the existing leading-instanton selector
`CYAxiverse.generate.LQtilde` (Priority 2, confirmed to implement eq. 3.19's
"reduced charge matrix" selection --
`validation/fuzzy_axions_2412_12012_leading_instanton_selector_20260818.md`).

Uses `pq_canonical_frame`'s single-leading-instanton approximation, not full
Hessian diagonalization: the source paper explicitly reserves the latter for
one hand-picked degenerate example (Sec. 4.2.6, footnote 26) and states the
hierarchical single-instanton formula (3.18)-(3.19) is what the bulk
Algorithm-1 ensemble uses (Sec. 4.3).
"""

const FUZZY_AXION_MASS_TARGET_EV = 1e-18
const FUZZY_AXION_QCD_VOLUME_MIN = 25.0
const FUZZY_AXION_QCD_VOLUME_MAX = 40.0

"""
    leading_axion_reference_data(Q, tau, cy_volume, prefactor_P, gravitino_mass_planck_units, inverse_metric)

Select the up-to-h11 leading (dominant) instantons via `LQtilde` and return,
for each, its raw prime-divisor volume `tau_a(t0)` and physical mass
`m_a(t0)` in eV -- the two inputs the closed-form λ_a solver needs (scope
note Sec. 1.4).

`Q` is the h11×N GLSM charge matrix (N = h11+4 prime toric divisors,
instanton charges as columns); `tau` is the length-N vector of prime-divisor
volumes at t0 (both from the Priority-1 Python export). `prefactor_P` and
`gravitino_mass_planck_units` are Phase 1's `fuzzy_axion_prefactor_P` output
and `fuzzy_axion_gravitino_mass(...; mplanck_ev=1.0)` output respectively
(the latter passed in Planck units, i.e. m_3/2 measured as a fraction of
`Mpl`, matching eq. 3.18's dimensionless Kähler-potential-frame quantities
before the final Mpl-to-eV "restoration").

`inverse_metric` is `K^{ij}` at t0 (the Priority-1 export's `inverse_metric`).

Also returns `divisor_index`, the original 1-based column of `Q`/`tau` each
selected leading axion was built from -- needed by
[`enumerate_fuzzy_axion_models`](@ref) to exclude an axion from being paired
with its own divisor as a candidate QCD divisor (see that function's
docstring).
"""
function leading_axion_reference_data(Q::AbstractMatrix{Int}, tau::AbstractVector{<:Real},
        cy_volume::Real, prefactor_P::Real, gravitino_mass_planck_units::Real,
        inverse_metric::AbstractMatrix{<:Real})
    cy_volume > 0 || throw(ArgumentError("cy_volume must be positive"))
    prefactor_P > 0 || throw(ArgumentError("prefactor_P must be positive"))
    gravitino_mass_planck_units > 0 ||
        throw(ArgumentError("gravitino_mass_planck_units must be positive"))
    size(Q, 1) == size(inverse_metric, 1) == size(inverse_metric, 2) ||
        throw(DimensionMismatch("Q's row count must match inverse_metric's square dimension"))
    size(Q, 2) == length(tau) ||
        throw(DimensionMismatch("Q must have one column per prime-divisor volume in tau"))

    Kinv = Matrix{Float64}(inverse_metric)
    Kinv = (Kinv + Kinv') / 2
    C = Matrix(cholesky(Symmetric(Kinv)).L)

    # eq. 3.18's constant prefactor (identical for every instanton at the
    # reference dilation λ=1): log10(8π * sqrt(P) * (m_3/2/Mpl) / V).
    # `instanton_scales` builds only log10[(Q·τ) exp(-2πQ·τ)]; this restores
    # the missing physical prefactor before the leading-instanton selection
    # and mass calculation use it.
    log10_prefactor = log10(8π) + 0.5 * log10(Float64(prefactor_P)) +
        log10(Float64(gravitino_mass_planck_units)) - log10(Float64(cy_volume))

    L = instanton_scales(Float64.(tau), 1.0)
    L[2, :] .+= log10_prefactor

    selected = LQtilde(Matrix{Int}(Q), L)
    Qleading = Matrix(selected.Qtilde') * C
    _, mapprox, _ = pq_canonical_frame(Qleading, selected.Ltilde)
    masses_log10_ev = mapprox .+ 9 .+ Float64(log10(constants()["MPlanck"])) .+
        Float64(constants()["log2π"])

    n_leading = size(selected.Qtilde, 2)
    tau_reference = zeros(Float64, n_leading)
    divisor_index = zeros(Int, n_leading)
    for a in 1:n_leading
        column = @view selected.Qtilde[:, a]
        match = findfirst(j -> @view(Q[:, j]) == column, axes(Q, 2))
        match === nothing && throw(ArgumentError(
            "a selected leading charge column was not found among the original Q columns"))
        tau_reference[a] = tau[match]
        divisor_index[a] = match
    end

    (; Qtilde = selected.Qtilde, tau_reference, divisor_index,
       mass_log10_ev_reference = masses_log10_ev)
end

"""
    fuzzy_axion_dilation_root(mass_reference_ev, tau_reference; mass_target_ev=FUZZY_AXION_MASS_TARGET_EV)

Closed-form λ solving `m_a(λ*t0) == mass_target_ev` exactly (arXiv:2412.12012
Sec. 3.4, eq. 3.24-3.27; derivation recorded in
`validation/fuzzy_axions_2412_12012_kaehler_qcd_model_count_scope_20260818.md`
Sec. 1.4):

    λ = sqrt(1 + log(mass_reference_ev / mass_target_ev) / (π * tau_reference))

Returns `nothing` when no positive real root exists (`mass_reference_ev <=
mass_target_ev`: dilating outward only decreases the mass further, so the
target can never be reached by increasing λ from the t0 reference).

Delegates to [`fuzzy_axion_dilation_root_log10`](@ref); prefer that form
directly when the reference mass is only available as `log10(mass)` (as
`leading_axion_reference_data` returns it) -- for a sufficiently suppressed
sub-leading instanton under the eq. 3.20 hierarchy, `10.0^log10_mass`
underflows to exactly `0.0` in `Float64` (observed live on a real h11=4
export record: `log10(mass) ≈ -390` for a fourth-ranked leading axion, well
below `Float64`'s ~1e-308 smallest positive value), which would make this
linear-mass form reject an otherwise well-defined contracting root.
"""
function fuzzy_axion_dilation_root(mass_reference_ev::Real, tau_reference::Real;
        mass_target_ev::Real=FUZZY_AXION_MASS_TARGET_EV)
    mass_reference_ev > 0 || throw(ArgumentError("mass_reference_ev must be positive"))
    fuzzy_axion_dilation_root_log10(log10(Float64(mass_reference_ev)), tau_reference;
        mass_target_ev)
end

"""
    fuzzy_axion_dilation_root_log10(mass_reference_log10_ev, tau_reference; mass_target_ev=FUZZY_AXION_MASS_TARGET_EV)

Same closed-form root as [`fuzzy_axion_dilation_root`](@ref)
(`λ = sqrt(1 + ln(mass_reference_ev/mass_target_ev) / (π*tau_reference))`,
rewritten with `ln(a/b) = ln(10)*(log10(a) - log10(b))`), but takes the
reference mass as `log10(mass_reference_ev)` directly. Never materializes
the linear reference mass, so it cannot underflow the way
`fuzzy_axion_dilation_root(10.0^mass_reference_log10_ev, ...)` can for a
strongly-suppressed sub-leading instanton -- see that function's docstring.
"""
function fuzzy_axion_dilation_root_log10(mass_reference_log10_ev::Real, tau_reference::Real;
        mass_target_ev::Real=FUZZY_AXION_MASS_TARGET_EV)
    isfinite(mass_reference_log10_ev) ||
        throw(ArgumentError("mass_reference_log10_ev must be finite"))
    tau_reference > 0 || throw(ArgumentError("tau_reference must be positive"))
    mass_target_ev > 0 || throw(ArgumentError("mass_target_ev must be positive"))
    argument = 1.0 + log(10.0) *
        (Float64(mass_reference_log10_ev) - log10(Float64(mass_target_ev))) /
        (π * Float64(tau_reference))
    argument > 0 || return nothing
    sqrt(argument)
end

"""
    fuzzy_axion_criterion_one(tau, lambda)

Criterion 1 (Sec. 3.3): every prime-toric-divisor volume (all h1,1+4 of
them) is >= 1 at the dilated point λ*t0. Uses eq. 3.25's homogeneous
`τ ↦ λ²τ` scaling directly on the Priority-1 export's raw
`prime_divisor_volumes`.
"""
function fuzzy_axion_criterion_one(tau::AbstractVector{<:Real}, lambda::Real)
    lambda > 0 || throw(ArgumentError("lambda must be positive"))
    all(>=(1.0), (lambda^2) .* tau)
end

"""
    fuzzy_axion_criterion_two(tau_qcd, lambda; volume_min=FUZZY_AXION_QCD_VOLUME_MIN, volume_max=FUZZY_AXION_QCD_VOLUME_MAX)

Criterion 2 (Sec. 3.3): the candidate QCD divisor's volume lands in
`[25, 40]` at the dilated point.
"""
function fuzzy_axion_criterion_two(tau_qcd::Real, lambda::Real;
        volume_min::Real=FUZZY_AXION_QCD_VOLUME_MIN,
        volume_max::Real=FUZZY_AXION_QCD_VOLUME_MAX)
    lambda > 0 || throw(ArgumentError("lambda must be positive"))
    scaled = (lambda^2) * tau_qcd
    volume_min <= scaled <= volume_max
end

"""
    enumerate_fuzzy_axion_models(Q, tau, cy_volume, prefactor_P, gravitino_mass_planck_units, inverse_metric; kwargs...)

Enumerate every `(QCD divisor, fuzzy-axion)` model Algorithm 1 (Sec. 4.1)
accepts for one `(X, FRST)` orientifold candidate.

For each leading axion index `a` (up to h1,1 of them), the closed-form
`λ_a` is solved once (independent of the QCD-divisor choice), criterion 1
is checked once at `λ_a` (also QCD-divisor-independent), and then every one
of the h1,1+4 candidate QCD divisors is checked against criterion 2 at that
same `λ_a`. This is mathematically equivalent to Algorithm 1's literal
`for D ... for λ ...` nesting (criteria 1 and 3 do not depend on `D`) but
avoids re-solving `λ_a` once per divisor. One record is returned per
accepted `(D, a)` pair; criterion 3 is satisfied by construction at `λ_a`.

Each returned model carries `mass_reference_log10_ev` (`log10` of the
reference-point mass), not the linear value -- the linear value can
underflow to exactly `0.0` for a strongly-suppressed sub-leading instanton
(see [`fuzzy_axion_dilation_root`](@ref)'s docstring), so the root-solve
itself uses [`fuzzy_axion_dilation_root_log10`](@ref) directly on
`reference.mass_log10_ev_reference`, never materializing the linear mass.

A candidate QCD divisor `D` equal to axion `a`'s own divisor (`D ==
reference.divisor_index[a]`) is excluded: that would require the same
four-cycle to simultaneously host the D7-brane stack realizing QCD
(criterion 2) and be the instanton wrapping that generates axion `a`'s own
ultralight mass (criterion 3) -- every worked example in the source
(Sec. 4.2) keeps the QCD divisor and the fuzzy-axion divisor distinct
(e.g. Sec. 4.2.1: "the QCD axion is associated to the prime toric divisor
D6, and the fuzzy axion is associated to the prime toric divisor D2"), and
this pairing is not merely disfavored but physically incoherent (the same
cycle cannot be both the fixed engineering choice hosting a non-dynamical
gauge stack and the field being continuously dilated toward an unrelated
mass target). Confirmed empirically to matter: on the real Algorithm-1
canonical-tip point for the paper's own h1,1=2 example, every one of this
geometry's currently-generated models was of exactly this self-referential
form before this exclusion was added (see
`validation/fuzzy_axions_2412_12012_mass_formula_missing_qcd_axion_20260818.md`).
"""
function enumerate_fuzzy_axion_models(Q::AbstractMatrix{Int}, tau::AbstractVector{<:Real},
        cy_volume::Real, prefactor_P::Real, gravitino_mass_planck_units::Real,
        inverse_metric::AbstractMatrix{<:Real};
        mass_target_ev::Real=FUZZY_AXION_MASS_TARGET_EV,
        qcd_volume_min::Real=FUZZY_AXION_QCD_VOLUME_MIN,
        qcd_volume_max::Real=FUZZY_AXION_QCD_VOLUME_MAX)
    reference = leading_axion_reference_data(Q, tau, cy_volume, prefactor_P,
        gravitino_mass_planck_units, inverse_metric)
    n_leading = length(reference.tau_reference)
    n_divisors = length(tau)
    models = NamedTuple[]
    for a in 1:n_leading
        mass_reference_log10_ev = reference.mass_log10_ev_reference[a]
        lambda = fuzzy_axion_dilation_root_log10(mass_reference_log10_ev, reference.tau_reference[a];
            mass_target_ev=mass_target_ev)
        lambda === nothing && continue
        fuzzy_axion_criterion_one(tau, lambda) || continue
        for divisor_index in 1:n_divisors
            divisor_index == reference.divisor_index[a] && continue
            fuzzy_axion_criterion_two(tau[divisor_index], lambda;
                volume_min=qcd_volume_min, volume_max=qcd_volume_max) || continue
            push!(models, (;
                axion_index=a,
                qcd_divisor_index=divisor_index,
                lambda=lambda,
                mass_reference_log10_ev=mass_reference_log10_ev,
                tau_reference=reference.tau_reference[a],
            ))
        end
    end
    models
end
