"""Source-approximation helpers for the Appendix-B BHSR likelihood.

These functions implement the projected one-dimensional Gaussian method and
probability products. The source Table I does not define a complete set of
Gaussian sigmas, so they do not manufacture the unavailable source-wide result.
"""

if !isdefined(@__MODULE__, :BHSR_SI)
    include(joinpath(@__DIR__, "bhsr_regge_reconstruction.jl"))
end

const BHSR_APPENDIX_B_ROUTE = "REFERENCE_2018_APPENDIX_B"
const BHSR_APPENDIX_B_EXPECTED_ROWS = 24
const BHSR_APPENDIX_B_UNRESOLVED_ROWS = (
    "Cygnus X-1", "GRS 1915+105", "NGC 3783", "MCG-6-30-15", "Mrk 110", "NGC 4051",
)

function _bhsr_contour_value(f, x::BigFloat)
    value = f(x)
    ismissing(value) && return missing
    value isa Real || throw(ArgumentError("contour function must return a real value or missing"))
    isfinite(value) || return missing
    return _bhsr_big(value)
end

function _bhsr_erfc(x::BigFloat)
    y = BigFloat(0)
    ccall((:mpfr_erfc, Base.MPFR.libmpfr), Int32,
          (Ref{BigFloat}, Ref{BigFloat}, Base.MPFR.MPFRRoundingMode),
          y, x, Base.MPFR.ROUNDING_MODE[])
    return y
end

"Standard normal CDF evaluated by MPFR at the active BigFloat precision."
function bhsr_standard_normal_cdf(z::Real; precision_bits::Integer = 256)
    return setprecision(BigFloat, precision_bits) do
        x = _bhsr_big(z) / sqrt(BigFloat(2))
        x < 0 ? _bhsr_erfc(-x) / 2 : 1 - _bhsr_erfc(x) / 2
    end
end

"Five-point centered derivative, with a second-order one-sided boundary rule."
function bhsr_contour_derivative(f, x::Real; lower::Real = -Inf,
                                  upper::Real = Inf,
                                  step_scale::Real = big"1e-5")
    xb, lo, hi = _bhsr_big(x), _bhsr_big(lower), _bhsr_big(upper)
    h = _bhsr_big(step_scale) * max(BigFloat(1), abs(xb))
    h > 0 || throw(ArgumentError("derivative step must be positive"))
    if xb - 2h >= lo && xb + 2h <= hi
        values = map(x -> _bhsr_contour_value(f, x), (xb - 2h, xb - h, xb + h, xb + 2h))
        any(ismissing, values) && return missing
        return (values[1] - 8values[2] + 8values[3] - values[4]) / (12h)
    elseif xb - 2h < lo
        xb + 2h <= hi || throw(ArgumentError("support is too narrow for a one-sided derivative"))
        values = map(x -> _bhsr_contour_value(f, x), (xb, xb + h, xb + 2h))
        any(ismissing, values) && return missing
        return (-3values[1] + 4values[2] - values[3]) / (2h)
    else
        xb - 2h >= lo || throw(ArgumentError("support is too narrow for a one-sided derivative"))
        values = map(x -> _bhsr_contour_value(f, x), (xb, xb - h, xb - 2h))
        any(ismissing, values) && return missing
        return (3values[1] - 4values[2] + values[3]) / (2h)
    end
end

function bhsr_projected_sigma_y(sigma_x::Real, sigma_y::Real, derivative::Real)
    sx, sy, fp = _bhsr_big(sigma_x), _bhsr_big(sigma_y), _bhsr_big(derivative)
    sx > 0 && sy > 0 || throw(ArgumentError("measurement sigmas must be positive"))
    return sqrt(sy^2 + fp^2 * sx^2)
end

function bhsr_projected_sigma_x(sigma_x::Real, sigma_y::Real, inverse_derivative::Real)
    sx, sy, gp = _bhsr_big(sigma_x), _bhsr_big(sigma_y), _bhsr_big(inverse_derivative)
    sx > 0 && sy > 0 || throw(ArgumentError("measurement sigmas must be positive"))
    return sqrt(sx^2 + gp^2 * sy^2)
end

"Eq. (97): allowed probability for a contour y=f(x) defined at xbar."
function bhsr_allowed_probability_direct(xbar::Real, ybar::Real,
                                          sigma_x::Real, sigma_y::Real,
                                          f; lower::Real = -Inf,
                                          upper::Real = Inf)
    xb, yb = _bhsr_big(xbar), _bhsr_big(ybar)
    lo, hi = _bhsr_big(lower), _bhsr_big(upper)
    if xb < lo || xb > hi
        return (; route_identity = BHSR_APPENDIX_B_ROUTE,
                projection = :y_of_x,
                status = :outside_contour_support,
                derivative = missing,
                sigma_effective = missing,
                probability_allowed = missing)
    end
    fbar = _bhsr_contour_value(f, xb)
    if ismissing(fbar)
        return (; route_identity = BHSR_APPENDIX_B_ROUTE,
                projection = :y_of_x,
                status = :missing_contour_value,
                derivative = missing,
                sigma_effective = missing,
                probability_allowed = missing)
    end
    derivative = bhsr_contour_derivative(f, xb; lower, upper)
    if ismissing(derivative)
        return (; route_identity = BHSR_APPENDIX_B_ROUTE,
                projection = :y_of_x,
                status = :incomplete_derivative_support,
                derivative = missing,
                sigma_effective = missing,
                probability_allowed = missing)
    end
    sigma_eff = bhsr_projected_sigma_y(sigma_x, sigma_y, derivative)
    probability = bhsr_standard_normal_cdf((fbar - yb) / sigma_eff)
    return (; route_identity = BHSR_APPENDIX_B_ROUTE,
            projection = :y_of_x,
            status = :evaluated,
            derivative,
            sigma_effective = sigma_eff,
            probability_allowed = probability)
end

"""Eq. (98) inverse branch treatment: nearest branch sets sigma_x; the
probability between both inverse roots is disallowed."""
function bhsr_allowed_probability_inverse(xbar::Real, sigma_x::Real,
                                           sigma_y::Real,
                                           inverse_roots::AbstractVector,
                                           inverse_derivatives::AbstractVector)
    length(inverse_roots) == length(inverse_derivatives) ||
        throw(DimensionMismatch("each inverse branch needs its derivative"))
    isempty(inverse_roots) && throw(ArgumentError("at least one inverse branch is required"))
    length(inverse_roots) <= 2 ||
        throw(ArgumentError("Appendix-B inverse treatment is specified for at most two branches"))
    any(ismissing, inverse_roots) && throw(ArgumentError("inverse roots must be defined on source support"))
    any(ismissing, inverse_derivatives) && throw(ArgumentError("inverse branch derivative is unavailable"))
    all(x -> x isa Real, inverse_roots) || throw(ArgumentError("inverse roots must be real"))
    all(x -> x isa Real && isfinite(x), inverse_derivatives) ||
        throw(ArgumentError("inverse branch derivatives must be finite and real"))
    roots = map(_bhsr_big, inverse_roots)
    slopes = map(_bhsr_big, inverse_derivatives)
    length(unique(roots)) == length(roots) ||
        throw(ArgumentError("coincident inverse roots are ambiguous at a contour cusp"))
    xb = _bhsr_big(xbar)
    distances = abs.(roots .- xb)
    nearest_distance = minimum(distances)
    count(==(nearest_distance), distances) == 1 ||
        throw(ArgumentError("nearest inverse branch is ambiguous at xbar"))
    nearest = findfirst(==(nearest_distance), distances)
    sigma_eff = bhsr_projected_sigma_x(sigma_x, sigma_y, slopes[nearest])
    low, high = extrema(roots)
    p_inside = bhsr_standard_normal_cdf((high - xb) / sigma_eff) -
        bhsr_standard_normal_cdf((low - xb) / sigma_eff)
    p_allowed = clamp(BigFloat(1) - p_inside, BigFloat(0), BigFloat(1))
    return (; route_identity = BHSR_APPENDIX_B_ROUTE,
            projection = :inverse_x,
            nearest_branch = nearest,
            inverse_derivative = slopes[nearest],
            sigma_effective = sigma_eff,
            probability_disallowed_between_branches = p_inside,
            probability_allowed = p_allowed)
end

"Union/envelope boundary from the per-mode exclusion thresholds at one mass."
function bhsr_union_boundary(values::AbstractVector)
    present = [_bhsr_big(value) for value in values if !ismissing(value)]
    return isempty(present) ? missing : minimum(present)
end

"Return the total union boundary function, including finite mode supports."
function bhsr_union_boundary_function(contours::AbstractVector)
    isempty(contours) && throw(ArgumentError("at least one mode contour is required"))
    return x -> bhsr_union_boundary([contour(x) for contour in contours])
end

function _bhsr_probability_product(values)
    isempty(values) && throw(ArgumentError("probability products require nonempty inputs"))
    any(ismissing, values) && return missing
    probabilities = _bhsr_big.(values)
    all(p -> 0 <= p <= 1, probabilities) ||
        throw(DomainError(probabilities, "allowed probabilities must lie in [0,1]"))
    return prod(probabilities)
end

"Eq. (96), then the governed product over axions; missing terms fail closed."
function bhsr_probability_tree(probabilities_by_axion_bh::AbstractVector)
    isempty(probabilities_by_axion_bh) &&
        throw(ArgumentError("at least one source-relevant axion is required"))
    single_axion_allowed = [_bhsr_probability_product(values)
                            for values in probabilities_by_axion_bh]
    if any(ismissing, single_axion_allowed)
        return (; route_identity = BHSR_APPENDIX_B_ROUTE,
                single_axion_allowed,
                geometry_allowed = missing,
                geometry_excluded = missing,
                threshold_exceeded = missing)
    end
    geometry_allowed = _bhsr_probability_product(single_axion_allowed)
    geometry_excluded = BigFloat(1) - geometry_allowed
    return (; route_identity = BHSR_APPENDIX_B_ROUTE,
            single_axion_allowed,
            geometry_allowed,
            geometry_excluded,
            threshold_exceeded = geometry_excluded > big"0.9545")
end

"Fail-closed audit of source Table-I uncertainties; never infers sigma."
function bhsr_sigma_gate(rows; expected_rows::Integer = BHSR_APPENDIX_B_EXPECTED_ROWS)
    count = length(rows)
    if count != expected_rows
        return (; route_identity = BHSR_APPENDIX_B_ROUTE,
                status = :unavailable,
                expected_rows,
                received_rows = count,
                unresolved = String[])
    end
    unresolved = String[]
    for row in rows
        name = String(getproperty(row, :name))
        sigma_mass = getproperty(row, :sigma_mass)
        sigma_spin = getproperty(row, :sigma_spin)
        mass_source = hasproperty(row, :sigma_mass_source) ?
            getproperty(row, :sigma_mass_source) : missing
        spin_source = hasproperty(row, :sigma_spin_source) ?
            getproperty(row, :sigma_spin_source) : missing
        complete = !ismissing(sigma_mass) && !ismissing(sigma_spin) &&
            isequal(mass_source, "source_defined") &&
            isequal(spin_source, "source_defined")
        complete || push!(unresolved, name)
    end
    return (; route_identity = BHSR_APPENDIX_B_ROUTE,
            status = isempty(unresolved) ? :complete : :unavailable,
            expected_rows,
            received_rows = count,
            unresolved)
end

"The source audit found no complete source-defined Table-I sigma set."
bhsr_sourcewide_likelihood_status() =
    (; route_identity = BHSR_APPENDIX_B_ROUTE,
       status = :unavailable,
       source_rows = BHSR_APPENDIX_B_EXPECTED_ROWS,
       unresolved_censored_spin_rows = collect(BHSR_APPENDIX_B_UNRESOLVED_ROWS),
       reason = "no source-defined conversion for censored or mixed-confidence errors")
