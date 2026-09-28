"""Formula-level reconstruction of the scalar analytic Regge curves in Fig. 3.

All numerical inputs are evaluated with BigFloat. Masses are in solar masses,
scalar masses in eV, and returned rates in s^-1. This implements the source's
analytic approximation; it does not claim a digitized Fig. 3 comparison.
"""

const BHSR_SI = (
    G = "6.67430e-11",
    c = "299792458",
    hbar = "1.054571817e-34",
    eV_J = "1.602176634e-19",
    M_sun = "1.98847e30",
    year_s = "31557600",
)

_bhsr_big(x::BigFloat) = x
_bhsr_big(x::AbstractFloat) = parse(BigFloat, string(x))
_bhsr_big(x::Real) = BigFloat(x)

struct BHSRMode
    n_r::Int
    l::Int
    m::Int
    label::String
    function BHSRMode(n_r::Integer, l::Integer, m::Integer, label::AbstractString)
        n_r >= 0 || throw(ArgumentError("radial overtone n_r must be nonnegative"))
        l >= 1 || throw(ArgumentError("scalar angular number l must be positive"))
        1 <= m <= l || throw(ArgumentError("this reconstruction requires 1 <= m <= l"))
        new(Int(n_r), Int(l), Int(m), String(label))
    end
end

"The frozen nodeless family n_r=0, m=l, with principal number N=l+1."
bhsr_nodeless_modes() = [BHSRMode(0, l, l, "|$(l + 1)$(l)$(l)>") for l in 1:5]

bhsr_principal_number(mode::BHSRMode) = mode.n_r + mode.l + 1

function _bhsr_constants()
    (; (k => parse(BigFloat, v) for (k, v) in pairs(BHSR_SI))...)
end

"Dimensionless gravitational coupling alpha = G M mu/(hbar c^3), with mu in eV."
function bhsr_alpha(mass_solar::Real, mu_eV::Real; precision_bits::Integer = 256)
    mass_solar > 0 || throw(ArgumentError("black-hole mass must be positive"))
    mu_eV > 0 || throw(ArgumentError("scalar mass must be positive"))
    return setprecision(BigFloat, precision_bits) do
        k = _bhsr_constants()
        mass_kg = _bhsr_big(mass_solar) * k.M_sun
        mu_joule = _bhsr_big(mu_eV) * k.eV_J
        k.G * mass_kg * mu_joule / (k.hbar * k.c^3)
    end
end

"Occupation needed to extract the specified spin change, using source Eq. (9)."
function bhsr_nmax(mass_solar::Real, mode::BHSRMode; delta_a::Real = big"0.1",
                   precision_bits::Integer = 256)
    mass_solar > 0 || throw(ArgumentError("black-hole mass must be positive"))
    delta_a > 0 || throw(ArgumentError("delta_a must be positive"))
    return setprecision(BigFloat, precision_bits) do
        k = _bhsr_constants()
        mass_kg = _bhsr_big(mass_solar) * k.M_sun
        k.G * mass_kg^2 * _bhsr_big(delta_a) /
            (BigFloat(mode.m) * k.hbar * k.c)
    end
end

function _bhsr_A(mode::BHSRMode)
    n = mode.n_r
    l = mode.l
    top = BigFloat(2)^(4l + 2) * BigFloat(factorial(big(2l + n + 1)))
    top /= BigFloat(l + n + 1)^(2l + 4) * BigFloat(factorial(big(n)))
    ratio = BigFloat(factorial(big(l))) /
        (BigFloat(factorial(big(2l))) * BigFloat(factorial(big(2l + 1))))
    return top * ratio^2
end

"Signed analytic scalar rate in s^-1 from source Eqs. (12)-(14)."
function bhsr_rate_s(mass_solar::Real, mu_eV::Real, spin::Real, mode::BHSRMode;
                     precision_bits::Integer = 256)
    mass_solar > 0 || throw(ArgumentError("black-hole mass must be positive"))
    mu_eV > 0 || throw(ArgumentError("scalar mass must be positive"))
    0 <= spin <= 1 || throw(ArgumentError("dimensionless spin must lie in [0,1]"))
    return setprecision(BigFloat, precision_bits) do
        k = _bhsr_constants()
        a = _bhsr_big(spin)
        mass_kg = _bhsr_big(mass_solar) * k.M_sun
        alpha = k.G * mass_kg * _bhsr_big(mu_eV) * k.eV_J /
            (k.hbar * k.c^3)
        N = bhsr_principal_number(mode)
        rplus = 1 + sqrt(1 - a^2)
        omega_bar = alpha * (1 - alpha^2 / (2 * BigFloat(N)^2))
        omega_h_bar = a / (2rplus)
        detuning_bar = BigFloat(mode.m) * omega_h_bar - omega_bar
        xlm = prod(
            BigFloat(j^2) * (1 - a^2) +
                4rplus^2 * (BigFloat(mode.m) * omega_bar - alpha)^2
            for j in 1:mode.l
        )
        gamma_m = 2alpha * rplus * detuning_bar * alpha^(4mode.l + 4) *
            _bhsr_A(mode) * xlm
        t_m_seconds = k.G * mass_kg / k.c^3
        gamma_m / t_m_seconds
    end
end

"Residual of Gamma_SR*tau >= log(Nmax); nonnegative means the mode grows enough."
function bhsr_free_field_residual(mass_solar::Real, mu_eV::Real, spin::Real,
                                  mode::BHSRMode, tau_years::Real;
                                  delta_a::Real = 0.1,
                                  precision_bits::Integer = 256)
    tau_years > 0 || throw(ArgumentError("black-hole age must be positive"))
    return setprecision(BigFloat, precision_bits) do
        k = _bhsr_constants()
        nmax = bhsr_nmax(mass_solar, mode; delta_a, precision_bits)
        nmax > 1 || throw(DomainError(nmax, "source logarithmic threshold requires Nmax > 1"))
        gamma = bhsr_rate_s(mass_solar, mu_eV, spin, mode; precision_bits)
        gamma * _bhsr_big(tau_years) * k.year_s - log(nmax)
    end
end

"Solve the source's direct spin contour by bisection; return `missing` if absent."
function bhsr_critical_spin(mass_solar::Real, mu_eV::Real, mode::BHSRMode,
                            tau_years::Real; delta_a::Real = 0.1,
                            precision_bits::Integer = 256,
                            max_iterations::Integer = 140,
                            absolute_tolerance::Real = big"1e-40")
    precision_bits > 0 || throw(ArgumentError("precision_bits must be positive"))
    max_iterations > 0 || throw(ArgumentError("max_iterations must be positive"))
    atol = BigFloat(absolute_tolerance)
    atol > 0 || throw(ArgumentError("absolute_tolerance must be positive"))
    return setprecision(BigFloat, precision_bits) do
        f(a) = bhsr_free_field_residual(mass_solar, mu_eV, a, mode, tau_years;
                                        delta_a, precision_bits)
        lo = BigFloat(0)
        hi = BigFloat(1)
        flo = f(lo)
        fhi = f(hi)
        flo >= 0 && return lo
        fhi < 0 && return missing
        fhi == 0 && return hi
        for _ in 1:max_iterations
            hi - lo <= atol && break
            mid = (lo + hi) / 2
            if f(mid) >= 0
                hi = mid
            else
                lo = mid
            end
        end
        hi - lo <= atol || error("critical-spin bisection did not meet absolute_tolerance=$atol after $max_iterations iterations")
        (lo + hi) / 2
    end
end

"Per-mode direct roots and the union boundary (lowest spin threshold)."
function bhsr_regge_row(mass_solar::Real, mu_eV::Real, tau_years::Real,
                        modes::AbstractVector{BHSRMode} = bhsr_nodeless_modes(); kwargs...)
    roots = Pair{String,Union{Missing,BigFloat}}[]
    for mode in modes
        push!(roots, mode.label => bhsr_critical_spin(mass_solar, mu_eV, mode,
                                                       tau_years; kwargs...))
    end
    present = BigFloat[last(value) for value in roots if !ismissing(last(value))]
    union_spin = isempty(present) ? missing : minimum(present)
    return (; mass_solar = _bhsr_big(mass_solar), per_mode = roots, union_spin)
end

"Direct logarithmic mass grid specified by the frozen Fig. 3 method manifest."
function bhsr_regge_grid(mu_eV::Real, tau_years::Real;
                         mass_log10_min::Real = -1,
                         mass_log10_max::Real = 2,
                         points::Integer = 1201,
                         modes::AbstractVector{BHSRMode} = bhsr_nodeless_modes(),
                         kwargs...)
    points >= 2 || throw(ArgumentError("mass grid needs at least two points"))
    mass_log10_max > mass_log10_min || throw(ArgumentError("mass grid bounds are reversed"))
    return setprecision(BigFloat, get(kwargs, :precision_bits, 256)) do
        lower = BigFloat(mass_log10_min)
        width = BigFloat(mass_log10_max) - lower
        [bhsr_regge_row(BigFloat(10)^(lower + width * BigFloat(i - 1) / (points - 1)),
                        mu_eV, tau_years, modes; kwargs...) for i in 1:points]
    end
end

const BHSR_BOSENOVA_ROUTE = "REFERENCE_2021_BOSENOVA"
const BHSR_BOSENOVA_TIMESCALES_YEARS = ("1e10", "4.5e7", "4.5e6")

"Source Eq. (13) quantities, retaining the signed diagonal quartic as provenance."
function bhsr_bosenova_details(mass_solar::Real, mu_eV::Real, mode::BHSRMode,
                                lambda_iiii::Real; c_bose::Real = big"5",
                                reduced_planck_GeV::Real = big"2.435e18",
                                delta_a::Real = big"0.1",
                                precision_bits::Integer = 256)
    lambda = _bhsr_big(lambda_iiii)
    !iszero(lambda) || throw(DomainError(lambda, "lambda_iiii must be nonzero for finite f_pert"))
    c_bose > 0 || throw(ArgumentError("c_bose must be positive"))
    reduced_planck_GeV > 0 || throw(ArgumentError("reduced Planck mass must be positive"))
    return setprecision(BigFloat, precision_bits) do
        alpha = bhsr_alpha(mass_solar, mu_eV; precision_bits)
        alpha > 0 || throw(DomainError(alpha, "alpha must be positive"))
        N = bhsr_principal_number(mode)
        f_pert_GeV = (_bhsr_big(mu_eV) / BigFloat("1e9")) / sqrt(abs(lambda))
        n_bose = BigFloat("1e78") * _bhsr_big(c_bose) * BigFloat(N)^4 /
            alpha^3 * (_bhsr_big(mass_solar) / 10)^2 *
            (f_pert_GeV / _bhsr_big(reduced_planck_GeV))^2
        n_max = bhsr_nmax(mass_solar, mode; delta_a, precision_bits)
        return (; route_identity = BHSR_BOSENOVA_ROUTE,
                lambda_iiii_signed = lambda,
                lambda_iiii_magnitude = abs(lambda),
                principal_number = N,
                f_pert_GeV,
                n_bose,
                n_max,
                c_bose = _bhsr_big(c_bose),
                delta_a = _bhsr_big(delta_a),
                omitted_interactions = ("off_diagonal_quartics", "cubic_interactions"))
    end
end

"Residual for the source Eq. (26) self-interaction-modified spin-down condition."
function bhsr_bosenova_residual(mass_solar::Real, mu_eV::Real, spin::Real,
                                 mode::BHSRMode, lambda_iiii::Real,
                                 tau_years::Real; kwargs...)
    tau_years > 0 || throw(ArgumentError("black-hole age must be positive"))
    return setprecision(BigFloat, get(kwargs, :precision_bits, 256)) do
        details = bhsr_bosenova_details(mass_solar, mu_eV, mode, lambda_iiii; kwargs...)
        details.n_bose > 1 || throw(DomainError(details.n_bose, "source log threshold requires N_Bose > 1"))
        gamma = bhsr_rate_s(mass_solar, mu_eV, spin, mode;
                            precision_bits = get(kwargs, :precision_bits, 256))
        k = _bhsr_constants()
        gamma * _bhsr_big(tau_years) * k.year_s * (details.n_bose / details.n_max) -
            log(details.n_bose)
    end
end

"Strict Eq. (26) decision: equality is the transition and is not efficient."
bhsr_bosenova_efficient(residual::Real) = residual > 0

function bhsr_bosenova_critical_spin(mass_solar::Real, mu_eV::Real,
                                     mode::BHSRMode, lambda_iiii::Real,
                                     tau_years::Real; precision_bits::Integer = 256,
                                     max_iterations::Integer = 140,
                                     absolute_tolerance::Real = big"1e-40",
                                     kwargs...)
    return setprecision(BigFloat, precision_bits) do
        residual(a) = bhsr_bosenova_residual(mass_solar, mu_eV, a, mode,
                                             lambda_iiii, tau_years;
                                             precision_bits, kwargs...)
        lo, hi = BigFloat(0), BigFloat(1)
        flo, fhi = residual(lo), residual(hi)
        flo >= 0 && return lo
        fhi < 0 && return missing
        for _ in 1:max_iterations
            hi - lo <= BigFloat(absolute_tolerance) && break
            mid = (lo + hi) / 2
            if residual(mid) > 0
                hi = mid
            else
                lo = mid
            end
        end
        hi - lo <= BigFloat(absolute_tolerance) ||
            error("bosenova critical-spin bisection did not meet tolerance")
        (lo + hi) / 2
    end
end

"Per-mode self-interaction transitions and their source-mode union boundary."
function bhsr_bosenova_regge_row(mass_solar::Real, mu_eV::Real,
                                  lambda_iiii::Real, tau_years::Real,
                                  modes::AbstractVector{BHSRMode} = bhsr_nodeless_modes(); kwargs...)
    roots = Pair{String,Union{Missing,BigFloat}}[]
    for mode in modes
        root = bhsr_bosenova_critical_spin(mass_solar, mu_eV, mode,
                                            lambda_iiii, tau_years; kwargs...)
        push!(roots, mode.label => root)
    end
    present = BigFloat[last(pair) for pair in roots if !ismissing(last(pair))]
    union_spin = isempty(present) ? missing : minimum(present)
    return (; route_identity = BHSR_BOSENOVA_ROUTE,
            mass_solar = _bhsr_big(mass_solar),
            per_mode = roots,
            union_spin)
end
