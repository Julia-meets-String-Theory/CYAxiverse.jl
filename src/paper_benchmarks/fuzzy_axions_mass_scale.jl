"""
Scalar mass-scale formulas from arXiv:2412.12012 (Sheridan, Carta, Gendler,
Jain, Marsh, McAllister, Righi, Rogers, Schachner, "Fuzzy Axions and
Associated Relics").

Phase 1 of the priority-2/3 model-stage scoping in
`validation/fuzzy_axions_2412_12012_kaehler_qcd_model_count_scope_20260818.md`.
Implements only the scalar Kähler-potential/prefactor/gravitino-mass layer
(source eq. 3.10, 3.12, 3.15, 3.28-3.29, 3.18). It does not compute instanton
scales, decay constants, or axion masses (eq. 3.18's `Λ_α` and eq. 3.19's
`f_i`/`m_i`) -- those are Phase 2/3, and belong here too per the
CYTools/Julia architecture split recorded in the Phase 2+ handoff, not in
the Python geometry driver.

These functions operate purely on already-extracted scalars (CY volume,
string coupling, flux superpotential, instanton exponentials) -- none of
them touch a CYTools object, matching the package convention that anything
not strictly requiring CYTools downstream belongs in this Julia package
rather than in `scripts/`.
"""

"""
    fuzzy_axion_kahler_potential(cy_volume)

Kähler potential for the Kähler moduli, `K = -2 log(V)` (eq. 3.10).

`cy_volume` is the Calabi-Yau volume `V` in the same units as CYTools'
`evaluate_kaehler_point`'s `cy_volume` output (i.e. the volume computed from
the triple intersection numbers and the Kähler point, not a physical volume
in GeV^-3).
"""
function fuzzy_axion_kahler_potential(cy_volume::Real)
    cy_volume > 0 || throw(ArgumentError("cy_volume must be positive"))
    return -2.0 * log(Float64(cy_volume))
end

"""
    fuzzy_axion_prefactor_P(gs)

Order-of-magnitude estimate `P ≈ gs^4 / 128` (eq. 3.28-3.29).

This is an *approximation*, not an exact relation: the source derives it
from `e^{Kother} = gs / (128 * Ṽ)` (eq. 3.28, mirror volume `Ṽ`) together
with the scaling estimate `Ṽ ~ gs^-3` (eq. 3.29), both stated with `~`/`≈`,
not `=`, in the source. The source's own reference value is
`gs = 0.5 -> P ~ 5e-4`, "the value we have used in our main analysis" --
reproduced by `fuzzy_axion_prefactor_P(0.5) == 0.5^4 / 128 == 4.8828125e-4`.
"""
function fuzzy_axion_prefactor_P(gs::Real)
    gs > 0 || throw(ArgumentError("gs must be positive"))
    return Float64(gs)^4 / 128.0
end

"""
    fuzzy_axion_flux_superpotential(w0, instanton_terms=nothing)

Total superpotential `W = W0 + Σ_α A_α exp(-S_α)` (eq. 3.12).

The source sets `A_α = 1` for the entire paper ("In the rest of the paper
we will set `A_α = 1`"), so `instanton_terms` (when given) must already be
the raw `exp(-S_α)` values with no separate prefactor -- do not multiply by
an `A_α` before passing them in.

When `instanton_terms` is omitted (the default), this returns `W0` exactly.
This is an explicit, documented approximation, not a silent one: away from
the boundary of criterion 3 (eq. 3.18's mass target), the instanton-
correction terms are parametrically smaller than `W0` for any accepted
`(X, D, λ)` triple, since a term comparable in size to `W0` would itself be
of order the axion potential scale. Phase 3 (the closed-form λ solver) is
expected to supply `instanton_terms` explicitly once per-divisor instanton
actions are available, rather than relying on this default in the final
model count.
"""
function fuzzy_axion_flux_superpotential(w0::Number, instanton_terms=nothing)
    total = Complex{Float64}(w0)
    if instanton_terms !== nothing
        total += sum(Complex{Float64}.(instanton_terms))
    end
    return total
end

"""
    fuzzy_axion_gravitino_mass(prefactor, kahler_pot, superpotential; mplanck_ev)

Gravitino mass `m_3/2 = sqrt(P) * exp(K/2) * |W|` (eq. 3.18), in eV.

`prefactor`, `kahler_pot`, and `superpotential` are the dimensionless
Planck-unit quantities from [`fuzzy_axion_prefactor_P`](@ref),
[`fuzzy_axion_kahler_potential`](@ref), and
[`fuzzy_axion_flux_superpotential`](@ref) respectively; the result is
converted to eV by the reduced Planck mass already used by the package's
Julia mass-spectrum code (`CYAxiverse.generate.constants()["MPlanck"]`, GeV,
converted here to eV), matching the source's statement (around eq. 3.30)
that Planck-unit formulas need `Mpl` factors "restored" for a physical
value.
"""
function fuzzy_axion_gravitino_mass(
    prefactor::Real,
    kahler_pot::Real,
    superpotential::Number;
    mplanck_ev::Real=Float64(constants()["MPlanck"]) * 1e9,
)
    prefactor >= 0 || throw(ArgumentError("prefactor (P) must be non-negative"))
    magnitude = abs(Complex{Float64}(superpotential))
    return sqrt(Float64(prefactor)) * exp(Float64(kahler_pot) / 2.0) * magnitude * Float64(mplanck_ev)
end
