const _ZETA3_DECIMAL = "1.2020569031595942853997381615114499907649862923404988817922715553418382057863"

# Apéry's alternating series is evaluated with 32 guard bits; the alternating
# tail is bounded by the first omitted term.
function _zeta3(::Type{BigFloat})
    requested_precision = precision(BigFloat)
    setprecision(BigFloat, requested_precision + 32) do
        total = zero(BigFloat)
        n = 1
        while true
            central_binomial = binomial(big(2n), big(n))
            term = BigFloat(5) /
                (BigFloat(2) * BigFloat(n)^3 * BigFloat(central_binomial))
            iseven(n) && (term = -term)
            total += term
            abs(term) <= eps(BigFloat) * abs(total) && break
            n += 1
        end
        setprecision(BigFloat, requested_precision) do
            total + zero(BigFloat)
        end
    end
end

function _zeta3(::Type{T}) where {T<:AbstractFloat}
    setprecision(BigFloat, max(precision(BigFloat), 256)) do
        convert(T, parse(BigFloat, _ZETA3_DECIMAL))
    end
end

"""Explicit coordinate, unit, phase, and retained/frozen-field identity."""
struct ModelConvention
    coordinate_map::String
    axion_periodicity::String
    condensate_branch::String
    frame::String
    length_and_alpha_prime_units::String
    planck_normalization::String
    kcs_treatment::String
    active_fields::Tuple{Vararg{String}}
    frozen_fields::Tuple{Vararg{String}}
    flux_assumptions::Tuple{Vararg{String}}
    heavy_field_reduction::String
    source_convention::String

    function ModelConvention(coordinate_map::String, axion_periodicity::String,
            condensate_branch::String, frame::String, length_and_alpha_prime_units::String,
            planck_normalization::String, kcs_treatment::String,
            active_fields::Tuple{Vararg{String}}, frozen_fields::Tuple{Vararg{String}},
            flux_assumptions::Tuple{Vararg{String}}, heavy_field_reduction::String,
            source_convention::String)
        strings = (coordinate_map, axion_periodicity, condensate_branch, frame,
            length_and_alpha_prime_units, planck_normalization, kcs_treatment,
            heavy_field_reduction, source_convention)
        all(!isempty, strings) || throw(ArgumentError("model convention fields must be explicit"))
        new(coordinate_map, axion_periodicity, condensate_branch, frame,
            length_and_alpha_prime_units, planck_normalization, kcs_treatment,
            active_fields, frozen_fields, flux_assumptions, heavy_field_reduction,
            source_convention)
    end
end

function ModelConvention(; coordinate_map, axion_periodicity, condensate_branch,
        frame, length_and_alpha_prime_units, planck_normalization, kcs_treatment,
        active_fields, frozen_fields, flux_assumptions, heavy_field_reduction,
        source_convention)
    strings = (coordinate_map, axion_periodicity, condensate_branch, frame,
        length_and_alpha_prime_units, planck_normalization, kcs_treatment,
        heavy_field_reduction, source_convention)
    all(!isempty, strings) || throw(ArgumentError("model convention fields must be explicit"))
    ModelConvention(String(coordinate_map), String(axion_periodicity),
        String(condensate_branch), String(frame), String(length_and_alpha_prime_units),
        String(planck_normalization), String(kcs_treatment),
        Tuple(String.(active_fields)), Tuple(String.(frozen_fields)),
        Tuple(String.(flux_assumptions)), String(heavy_field_reduction),
        String(source_convention))
end

"""Independent switches for the declared potential contributions."""
struct ModelSwitches
    bbhl_correction_enabled::Bool
    np_linear_enabled::Bool
    np_quadratic_enabled::Bool
    uplift_enabled::Bool
end

ModelSwitches(; bbhl_correction_enabled=true, np_linear_enabled=true,
    np_quadratic_enabled=true, uplift_enabled=false) = ModelSwitches(
        bbhl_correction_enabled, np_linear_enabled, np_quadratic_enabled,
        uplift_enabled)

"""Explicitly supplied uplift function and its assumption identity."""
struct UpliftSpec{F}
    evaluate::F
    identity::String
    provenance::String
    function UpliftSpec(evaluate::F, identity::AbstractString,
            provenance::AbstractString) where {F}
        evaluate isa Function ||
            throw(ArgumentError("uplift evaluator must be a callable function"))
        all(!isempty, (identity, provenance)) ||
            throw(ArgumentError("uplift identity and provenance must be explicit"))
        _manifest_value_is_immutable(evaluate) ||
            throw(ArgumentError("uplift evaluator must not retain mutable state"))
        new{F}(evaluate, String(identity), String(provenance))
    end
end

"""Phase-0 retained Kähler potential with a generic integer charge matrix."""
struct KahlerModel{T<:AbstractFloat,U}
    w0_magnitude::FrozenScalar{T}
    theta0::FrozenScalar{T}
    gs::FrozenScalar{T}
    kcs::FrozenScalar{T}
    amplitudes::FrozenArray{T,1}
    actions::FrozenArray{T,1}
    phases::FrozenArray{T,1}
    charges::FrozenArray{BigInt,2} # rows label instantons; columns label basis divisors
    convention::ModelConvention
    switches::ModelSwitches
    uplift::U
    identity::String

    function KahlerModel{T,U}(w0_magnitude::FrozenScalar{T},
            theta0::FrozenScalar{T}, gs::FrozenScalar{T}, kcs::FrozenScalar{T},
            amplitudes::FrozenArray{T,1}, actions::FrozenArray{T,1},
            phases::FrozenArray{T,1}, charges::FrozenArray{BigInt,2},
            convention::ModelConvention, switches::ModelSwitches, uplift::U,
            identity::String) where {T<:AbstractFloat,U}
        wmag, theta, coupling, kconstant = _thaw_scalar(w0_magnitude),
            _thaw_scalar(theta0), _thaw_scalar(gs), _thaw_scalar(kcs)
        all(isfinite, (wmag, theta, coupling, kconstant)) &&
            all(isfinite, amplitudes) && all(isfinite, actions) && all(isfinite, phases) ||
            throw(ArgumentError("model parameters must be finite"))
        wmag >= zero(T) || throw(ArgumentError("|W0| must be nonnegative"))
        coupling > zero(T) || throw(ArgumentError("g_s must be positive"))
        all(>=(zero(T)), amplitudes) ||
            throw(ArgumentError("instanton amplitudes must be nonnegative"))
        all(>(zero(T)), actions) ||
            throw(ArgumentError("instanton actions must be positive"))
        length(amplitudes) == length(actions) == length(phases) == size(charges, 1) ||
            throw(DimensionMismatch("one amplitude, action, and phase is required per charge row"))
        !isempty(identity) || throw(ArgumentError("model identity must be nonempty"))
        (uplift === nothing || uplift isa UpliftSpec) ||
            throw(ArgumentError("uplift must be a validated UpliftSpec or nothing"))
        switches.uplift_enabled && !(uplift isa UpliftSpec) &&
            throw(ArgumentError("uplift cannot be enabled without a validated UpliftSpec"))
        new{T,U}(w0_magnitude, theta0, gs, kcs, amplitudes, actions, phases,
            charges, convention, switches, uplift, identity)
    end
end

function Base.getproperty(model::KahlerModel, name::Symbol)
    if name in (:w0_magnitude, :theta0, :gs, :kcs)
        return _thaw_scalar(getfield(model, name))
    end
    getfield(model, name)
end

_model_parameter_float_type(value::BigFloat) = BigFloat
_model_parameter_float_type(value::AbstractFloat) = typeof(value)
_model_parameter_float_type(::Rational{BigInt}) = BigFloat
_model_parameter_float_type(::Rational) = Float64
_model_parameter_float_type(::BigInt) = BigFloat
_model_parameter_float_type(::Integer) = Float64
_model_parameter_float_type(value::Real) = typeof(float(value))
_model_parameter_float_type(value) =
    throw(ArgumentError("model parameters must be real numbers; got $(typeof(value))"))

_model_parameter_precision(value::BigFloat) = precision(value)
_model_parameter_precision(value) = 0

function KahlerModel(; w0_magnitude, theta0, gs, kcs, amplitudes,
        actions, phases, charges, convention::ModelConvention,
        switches=ModelSwitches(), uplift=nothing, identity="CYAX-0191-source-model-v1")
    parameter_values = (w0_magnitude, theta0, gs, kcs,
        Tuple(amplitudes)..., Tuple(actions)..., Tuple(phases)...)
    T = promote_type((_model_parameter_float_type(value) for value in parameter_values)...)
    T <: AbstractFloat || throw(ArgumentError("model parameters must promote to an AbstractFloat"))
    input_precision = maximum(_model_parameter_precision, parameter_values; init=0)
    bits = input_precision == 0 ? precision(BigFloat) : input_precision
    construct() = _construct_kahler_model(T, w0_magnitude, theta0, gs, kcs,
        amplitudes, actions, phases, charges, convention, switches, uplift, identity)
    T === BigFloat ? setprecision(construct, BigFloat, bits) : construct()
end

function _construct_kahler_model(::Type{T}, w0_magnitude, theta0, gs, kcs,
        amplitudes, actions, phases, charges, convention, switches, uplift,
        identity) where {T<:AbstractFloat}
    amps, aa, ph = T.(amplitudes), T.(actions), T.(phases)
    Q = BigInt.(charges)
    length(amps) == length(aa) == length(ph) == size(Q, 1) ||
        throw(DimensionMismatch("one amplitude, action, and phase is required per charge row"))
    !isempty(identity) || throw(ArgumentError("model identity must be nonempty"))
    wmag, theta, coupling, kconstant = deepcopy(T(w0_magnitude)),
        deepcopy(T(theta0)), deepcopy(T(gs)), deepcopy(T(kcs))
    all(isfinite, (wmag, theta, coupling, kconstant)) &&
        all(isfinite, amps) && all(isfinite, aa) && all(isfinite, ph) ||
        throw(ArgumentError("model parameters must be finite"))
    wmag >= zero(T) || throw(ArgumentError("|W0| must be nonnegative"))
    coupling > zero(T) || throw(ArgumentError("g_s must be positive"))
    all(>=(zero(T)), amps) || throw(ArgumentError("instanton amplitudes must be nonnegative"))
    all(>(zero(T)), aa) || throw(ArgumentError("instanton actions must be positive"))
    (uplift === nothing || uplift isa UpliftSpec) ||
        throw(ArgumentError("uplift must be a validated UpliftSpec or nothing"))
    switches.uplift_enabled && !(uplift isa UpliftSpec) &&
        throw(ArgumentError("uplift cannot be enabled without a validated UpliftSpec"))
    KahlerModel{T,typeof(uplift)}(FrozenScalar(wmag), FrozenScalar(theta),
        FrozenScalar(coupling), FrozenScalar(kconstant),
        FrozenArray(amps), FrozenArray(aa), FrozenArray(ph), FrozenArray(Q),
        convention, switches, uplift, String(identity))
end

"""Construct the 2020 basis-divisor model, one instanton per basis divisor."""
function source_basis_model(n_moduli::Integer; w0_magnitude, theta0, gs, kcs,
        amplitudes, actions, phases, convention::ModelConvention,
        switches=ModelSwitches(), uplift=nothing,
        identity="CYAX-0191-2020-source-basis-v1")
    n_moduli > 0 || throw(ArgumentError("n_moduli must be positive"))
    length(amplitudes) == length(actions) == length(phases) == n_moduli ||
        throw(DimensionMismatch("the source-basis model requires one term per divisor"))
    KahlerModel(; w0_magnitude, theta0, gs, kcs, amplitudes, actions, phases,
        charges=Matrix{Int}(I, n_moduli, n_moduli), convention, switches,
        uplift, identity)
end

"""Return Q-weighted divisor and axion coordinates for the CYAxiverse extension."""
function charge_coordinates(charges::AbstractMatrix{<:Integer}, tau::AbstractVector,
        rho::AbstractVector)
    size(charges, 2) == length(tau) == length(rho) ||
        throw(DimensionMismatch("Q must have one column per divisor coordinate"))
    _with_model_precision(charges, tau, rho) do
        charge_times(values) = if all(value -> value isa Integer || value isa Rational, values)
            BigInt.(charges) * _widen_exact_array(values)
        else
            charges * values
        end
        (; tau=charge_times(tau), rho=charge_times(rho))
    end
end

"""Contract a retained metric or inverse metric into the charged-divisor basis."""
function charge_metric_contraction(charges::AbstractMatrix{<:Integer},
        metric::AbstractMatrix)
    size(metric, 1) == size(metric, 2) == size(charges, 2) ||
        throw(DimensionMismatch("Q and metric dimensions differ"))
    _with_model_precision(charges, metric) do
        if all(value -> value isa Integer || value isa Rational, metric)
            exact_charges = BigInt.(charges)
            exact_metric = _widen_exact_array(metric)
            exact_charges * exact_metric * exact_charges'
        else
            charges * metric * charges'
        end
    end
end

"""Status fields are deliberately separate; there is no aggregate stability flag."""
struct MetricAssessment{T}
    status::Symbol
    finite::Bool
    nonsingular::Bool
    positive_definite::Bool
    minimum_eigenvalue::Union{Nothing,T}
end

function _metric_assessment(metric::AbstractMatrix{T}) where {T<:Real}
    _with_model_precision(metric) do
        _metric_assessment_at_precision(metric)
    end
end

function _metric_assessment_at_precision(metric::AbstractMatrix{T}) where {T<:Real}
    finite = all(isfinite, metric)
    if !finite
        return MetricAssessment{T}(:FAIL, false, false, false, nothing)
    end
    symmetric = Symmetric((metric + metric') / 2)
    values = try
        eigvals(symmetric)
    catch
        T[]
    end
    if isempty(values)
        return MetricAssessment{T}(:FAIL, true, false, false, nothing)
    end
    nonsingular = all(x -> !iszero(x), values)
    positive = nonsingular && all(>(zero(T)), values)
    MetricAssessment{T}(positive ? :PASS : :FAIL, true, nonsingular,
        positive, minimum(values))
end

struct ModelEvaluation{T<:Real}
    value::T
    contributions::NamedTuple
    active_contributions::NamedTuple
    volume::T
    tau::Vector{T}
    rho::Vector{T}
    xi::T
    xihat::T
    xihat_over_two::T
    Y::T
    parent_metric::Matrix{T}
    # Frozen-heavy-field kinetic pullback: the parent metric's retained TT block.
    retained_kinetic_metric::Matrix{T}
    # Real coordinates are ordered (t,rho); the saxion block uses the tau(t) pullback.
    retained_real_kinetic_metric::Matrix{T}
    # TT block of the inverse full metric, used by the reduced potential.
    retained_inverse_metric::Matrix{T}
    parent_metric_assessment::MetricAssessment{T}
    retained_metric_assessment::MetricAssessment{T}
    imported_domain_status::Symbol
    model_identity::String
    geometry_identity::String
end

_model_precision(::Nothing) = 0
_model_precision(value::BigFloat) = precision(value)
_model_precision(value::FrozenScalar{BigFloat}) = getfield(value, :encoded)[1]
_model_precision(value::FrozenScalar) = 0
_model_precision(value::Pair) =
    max(_model_precision(first(value)), _model_precision(last(value)))
_model_precision(value::Tuple) =
    maximum((_model_precision(item) for item in value); init=0)
_model_precision(value::NamedTuple) =
    maximum((_model_precision(item) for item in values(value)); init=0)
_model_precision(value::AbstractArray) =
    maximum((_model_precision(item) for item in value); init=0)
_model_precision(value::CanonicalIntersectionTensor) =
    _model_precision(value.coefficients)
_model_precision(value::GeometryRecord) =
    max(_model_precision(value.intersections), _model_precision(value.domain_inequalities))
function _model_precision(value::KahlerModel)
    maximum(_model_precision(getfield(value, name)) for name in
        (:w0_magnitude, :theta0, :gs, :kcs, :amplitudes, :actions, :phases))
end
_model_precision(value) = 0

function _with_model_precision(f, values...)
    stored_precision = maximum(_model_precision, values; init=0)
    bits = stored_precision == 0 ? precision(BigFloat) : stored_precision
    setprecision(f, BigFloat, bits)
end

function _kahler_data(model::KahlerModel, geometry::GeometryRecord,
        t::AbstractVector; rho=nothing)
    n = geometry.intersections.n
    length(t) == n || throw(DimensionMismatch("wrong two-cycle vector length"))
    rho === nothing || length(rho) == n || throw(DimensionMismatch("wrong axion vector length"))
    R = promote_type(eltype(t), typeof(model.gs),
        rho === nothing ? typeof(model.gs) : eltype(rho),
        eltype(geometry.intersections.coefficients) <: BigFloat ? BigFloat : typeof(model.gs),
        eltype(geometry.domain_inequalities) <: BigFloat ? BigFloat : typeof(model.gs))
    R <: AbstractFloat || throw(ArgumentError("evaluation coordinates must be floating point"))
    _with_model_precision(model, geometry, t, rho) do
        _kahler_data_at_precision(model, geometry, t, R)
    end
end

function _kahler_data_at_precision(model::KahlerModel, geometry::GeometryRecord,
        t::AbstractVector, R::Type{<:AbstractFloat})
    n = geometry.intersections.n
    tt = R.(t)
    gs, kcs = R(model.gs), R(model.kcs)
    s = inv(gs)
    V = R(calabi_yau_volume(geometry, tt))
    tau = R.(divisor_volumes(geometry, tt))
    xi = -_zeta3(R) * R(geometry.euler_characteristic) /
        (R(2) * (R(2) * R(π))^3)
    xihat = xi * s * sqrt(s)
    xihat_half = xihat / R(2)
    correction = model.switches.bbhl_correction_enabled ? xihat_half : zero(R)
    Y = V + correction
    isfinite(V) && V > zero(R) || throw(DomainError(V, "Calabi–Yau volume must be positive and finite"))
    isfinite(Y) && Y > zero(R) || throw(DomainError(Y, "Y = Vcal + xihat/2 must be positive and finite"))
    J = R.(divisor_volume_jacobian(geometry, tt))
    det(J) != zero(R) || throw(DomainError(J, "divisor-volume Jacobian is singular"))
    Jinv = inv(J)
    ytau = tt ./ R(2)
    ytautau = Jinv ./ R(2)
    ys = model.switches.bbhl_correction_enabled ? (R(3) * correction) / (R(2) * s) : zero(R)
    yss = model.switches.bbhl_correction_enabled ? (R(3) * correction) / (R(4) * s^2) : zero(R)
    hessian = zeros(R, n + 1, n + 1)
    hessian[1, 1] = inv(s^2) - R(2) * (yss / Y - (ys / Y)^2)
    for i in 1:n
        hessian[1, i + 1] = hessian[i + 1, 1] =
            R(2) * ys * ytau[i] / Y^2
        for j in 1:n
            hessian[i + 1, j + 1] = -R(2) *
                (ytautau[i, j] / Y - ytau[i] * ytau[j] / Y^2)
        end
    end
    parent_metric = hessian ./ R(4)
    retained_kinetic_metric = Matrix(Symmetric(parent_metric[2:end, 2:end]))
    schur_complement_metric = retained_kinetic_metric -
        (parent_metric[2:end, 1:1] * parent_metric[1:1, 2:end]) / parent_metric[1, 1]
    schur_complement_metric = Matrix(Symmetric(
        (schur_complement_metric + schur_complement_metric') / R(2)))
    full_inverse_tt_metric = inv(schur_complement_metric)
    k_tau = -R(2) .* ytau ./ Y
    k_t = k_tau ./ R(2)
    K = kcs - log(R(2) * s) - R(2) * log(Y)
    (; R, tt, gs, kcs, s, V, tau, xi, xihat, xihat_half, Y,
       parent_metric, retained_kinetic_metric, schur_complement_metric,
       full_inverse_tt_metric, k_tau, k_t,
       expK=exp(K), domain_status=imported_domain_status(geometry, tt))
end

function _model_terms(model::KahlerModel, data, rho::AbstractVector)
    _with_model_precision(model, data, rho) do
        _model_terms_at_precision(model, data, rho)
    end
end

function _model_terms_at_precision(model::KahlerModel, data, rho::AbstractVector)
    n = length(data.tau)
    length(rho) == n || throw(DimensionMismatch("wrong axion vector length"))
    R = data.R
    size(model.charges, 2) == n || throw(DimensionMismatch("charge matrix has the wrong divisor dimension"))
    w0 = R(model.w0_magnitude) * complex(cos(R(model.theta0)), sin(R(model.theta0)))
    wn = zero(Complex{R})
    dw = zeros(Complex{R}, n)
    z_terms = Vector{Complex{R}}(undef, length(model.amplitudes))
    for a in eachindex(model.amplitudes)
        qtau = zero(R)
        qrho = zero(R)
        for i in 1:n
            charge = R(model.charges[a, i])
            qtau += charge * data.tau[i]
            qrho += charge * R(rho[i])
        end
        action = R(model.actions[a])
        phase = R(model.phases[a]) - action * qrho
        z = R(model.amplitudes[a]) * exp(complex(-action * qtau, phase))
        z_terms[a] = z
        wn += z
        for i in 1:n
            dw[i] -= action * R(model.charges[a, i]) * z
        end
    end
    d0 = data.k_t .* w0
    dn = dw + data.k_t .* wn
    (; w0, wn, z_terms, d0, dn)
end

function _potential_contributions(model::KahlerModel, geometry::GeometryRecord,
        data, terms, rho::AbstractVector)
    _with_model_precision(model, geometry, data, terms, rho) do
        _potential_contributions_at_precision(model, geometry, data, terms, rho)
    end
end

function _potential_contributions_at_precision(model::KahlerModel,
        geometry::GeometryRecord, data, terms, rho::AbstractVector)
    M = data.full_inverse_tt_metric
    R = data.R
    alpha = data.expK * (real(dot(terms.d0, M * terms.d0)) - R(3) * abs2(terms.w0))
    linear = data.expK * (R(2) * real(dot(terms.d0, M * terms.dn)) -
        R(6) * real(conj(terms.w0) * terms.wn))
    quadratic = data.expK * (real(dot(terms.dn, M * terms.dn)) -
        R(3) * abs2(terms.wn))
    uplift_value = zero(data.R)
    if model.switches.uplift_enabled
        uplift_value = data.R(model.uplift.evaluate((; geometry, t=data.tt,
            tau=data.tau, rho=data.R.(rho), volume=data.V, xihat=data.xihat)))
        isfinite(uplift_value) || throw(DomainError(uplift_value, "uplift returned a nonfinite value"))
    end
    contributions = (; alpha3=alpha, np_linear=linear,
        np_quadratic=quadratic, optional_uplift=uplift_value)
    switches = model.switches
    active = (; alpha3=switches.bbhl_correction_enabled ? alpha : zero(data.R),
        np_linear=switches.np_linear_enabled ? linear : zero(data.R),
        np_quadratic=switches.np_quadratic_enabled ? quadratic : zero(data.R),
        optional_uplift=switches.uplift_enabled ? uplift_value : zero(data.R))
    value = active.alpha3 + active.np_linear + active.np_quadratic + active.optional_uplift
    all(isfinite, (alpha, linear, quadratic, uplift_value, value)) ||
        throw(DomainError(value, "potential evaluation produced a nonfinite contribution"))
    (; contributions, active, value)
end

"""Scale stationarity by active-term magnitudes, with a no-scale fallback."""
function characteristic_potential_scale(model::KahlerModel,
        geometry::GeometryRecord, t::AbstractVector, rho::AbstractVector)
    _with_model_precision(model, geometry, t, rho) do
        data = _kahler_data(model, geometry, t; rho)
        data.domain_status === :PASS ||
            throw(DomainError(t, "two-cycle coordinates violate imported cone inequalities"))
        terms = _model_terms(model, data, rho)
        potential_parts = _potential_contributions(model, geometry, data, terms, rho)
        scale = sum(abs, values(potential_parts.active))
        if iszero(scale)
            superpotential_envelope = abs(terms.w0) + sum(abs, terms.z_terms)
            scale = abs(data.expK) * superpotential_envelope^2
            iszero(scale) && (scale = one(data.R))
        end
        isfinite(scale) && scale > zero(scale) ||
            throw(DomainError(scale, "the declared model has no finite positive characteristic potential scale"))
        scale
    end
end

"""Evaluate the declared potential, retaining the full heavy-field Schur block."""
function evaluate_potential(model::KahlerModel, geometry::GeometryRecord,
        t::AbstractVector, rho::AbstractVector; enforce_domain=true)
    _with_model_precision(model, geometry, t, rho) do
        data = _kahler_data(model, geometry, t; rho)
        enforce_domain && data.domain_status !== :PASS &&
            throw(DomainError(t, "two-cycle coordinates violate imported cone inequalities"))
        terms = _model_terms(model, data, rho)
        potential_parts = _potential_contributions(model, geometry, data, terms, rho)
        contributions, active, value = potential_parts.contributions, potential_parts.active,
            potential_parts.value
        n = length(data.tau)
        retained_real = zeros(data.R, 2n, 2n)
        tau_jacobian = data.R.(divisor_volume_jacobian(geometry, data.tt))
        retained_real[1:n, 1:n] .= data.R(2) .* (
            tau_jacobian' * data.retained_kinetic_metric * tau_jacobian)
        retained_real[(n + 1):end, (n + 1):end] .= data.R(2) .* data.retained_kinetic_metric
        all(isfinite, data.parent_metric) && all(isfinite, data.retained_kinetic_metric) &&
            all(isfinite, data.full_inverse_tt_metric) && all(isfinite, retained_real) ||
            throw(DomainError(retained_real, "potential evaluation produced a nonfinite kinetic metric"))
        ModelEvaluation(value, contributions, active, data.V, data.tau,
            data.R.(rho), data.xi, data.xihat, data.xihat_half, data.Y,
            data.parent_metric, data.retained_kinetic_metric, retained_real,
            data.full_inverse_tt_metric, _metric_assessment(data.parent_metric),
            _metric_assessment(retained_real), data.domain_status,
            model.identity, geometry.artifact_sha256)
    end
end

"""Full retained potential in the Julia convention `T=tau+i*rho`."""
potential(model::KahlerModel, geometry::GeometryRecord, t, rho; kwargs...) =
    evaluate_potential(model, geometry, t, rho; kwargs...).value

"""Adopted unequal-index phase: a_i rho_i - a_j rho_j - phi_i + phi_j."""
function quadratic_phase(action_i, rho_i, phase_i, action_j, rho_j, phase_j)
    _with_model_precision(action_i, rho_i, phase_i, action_j, rho_j, phase_j) do
        action_i * rho_i - action_j * rho_j - phase_i + phase_j
    end
end

function direct_complex_interference_phase(action_i, q_i, rho, phase_i,
        action_j, q_j, phase_j)
    _with_model_precision(action_i, q_i, rho, phase_i, action_j, q_j, phase_j) do
        z_i = exp(complex(zero(action_i), phase_i - action_i * dot(q_i, rho)))
        z_j = exp(complex(zero(action_j), phase_j - action_j * dot(q_j, rho)))
        angle(z_i * conj(z_j))
    end
end

"""Independent analytic axion derivative of the reduced potential."""
function analytic_axion_gradient(model::KahlerModel, geometry::GeometryRecord,
        t::AbstractVector, rho::AbstractVector)
    model.switches.uplift_enabled &&
        throw(ArgumentError("the supplied uplift has no declared analytic axion derivative"))
    _with_model_precision(model, geometry, t, rho) do
        data = _kahler_data(model, geometry, t; rho)
        data.domain_status === :PASS || throw(DomainError(t, "coordinates violate the imported cone"))
        terms = _model_terms(model, data, rho)
        n, na = length(data.tau), length(model.amplitudes)
        R = data.R
        grad = zeros(R, n)
        for j in 1:n
            dwn = zero(Complex{R})
            ddw = zeros(Complex{R}, n)
            for a in 1:na
                action = R(model.actions[a])
                qj = R(model.charges[a, j])
                z = terms.z_terms[a]
                dz = -im * action * qj * z
                dwn += dz
                for i in 1:n
                    ddw[i] += im * action^2 * R(model.charges[a, i]) * qj * z
                end
            end
            ddn = ddw + data.k_t .* dwn
            derivative_linear = R(2) * real(dot(terms.d0, data.full_inverse_tt_metric * ddn)) -
                R(6) * real(conj(terms.w0) * dwn)
            derivative_quadratic = R(2) * real(dot(terms.dn, data.full_inverse_tt_metric * ddn)) -
                R(6) * real(conj(terms.wn) * dwn)
            grad[j] = data.expK * ((model.switches.np_linear_enabled ? derivative_linear : zero(R)) +
                (model.switches.np_quadratic_enabled ? derivative_quadratic : zero(R)))
        end
        grad
    end
end
