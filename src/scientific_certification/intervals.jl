const _RD = RoundDown
const _RU = RoundUp
const _IR = Ref{Any}(nothing)
const _RATIONAL_COMPONENT_DIGITS = 4096
const _MAX_RATIONAL_COMPONENT_VALUE = big(10)^_RATIONAL_COMPONENT_DIGITS - 1
const _RATIONAL_COMPONENT_BITS = ndigits(_MAX_RATIONAL_COMPONENT_VALUE; base=2)
const _MAX_DYADIC_DENOMINATOR_EXPONENT = let k = _RATIONAL_COMPONENT_BITS
    while ndigits(big(1) << k; base=10) > _RATIONAL_COMPONENT_DIGITS
        k -= 1
    end
    k
end

struct ProofArithmeticError <: Exception
    message::String
end
Base.showerror(io::IO, e::ProofArithmeticError) = print(io, e.message)

_add(x::BigFloat, y::BigFloat, r::RoundingMode) = IA._add_round(_interval_rounding(), x, y, r)
_sub(x::BigFloat, y::BigFloat, r::RoundingMode) = IA._sub_round(_interval_rounding(), x, y, r)
_mul(x::BigFloat, y::BigFloat, r::RoundingMode) = IA._mul_round(_interval_rounding(), x, y, r)
_div(x::BigFloat, y::BigFloat, r::RoundingMode) = IA._div_round(_interval_rounding(), x, y, r)
_sin(x::BigFloat, r::RoundingMode) = IA._sin_round(_interval_rounding(), x, r)
_cos(x::BigFloat, r::RoundingMode) = IA._cos_round(_interval_rounding(), x, r)
_exp(x::BigFloat, r::RoundingMode) = IA._exp_round(_interval_rounding(), x, r)
_log(x::BigFloat, r::RoundingMode) = IA._log_round(_interval_rounding(), x, r)

"""Run proof work serially and restore the caller's BigFloat context."""
function _with_precision(f::Function, bits::Int)
    lock(_PROOF_LOCK) do
        setprecision(BigFloat, bits) do
            setrounding(BigFloat, RoundNearest) do
                f()
            end
        end
    end
end

_backend_supported() = Base.pkgversion(IA) == _BACKEND_VERSION

function _interval_rounding()
    rounding = _IR[]
    if rounding === nothing
        _require_backend()
        rounding = IA.IntervalRounding{:correct}()
        _IR[] = rounding
    end
    return rounding
end

function _require_backend()
    _backend_supported() || throw(BackendUnsupported("INTERVAL_BACKEND_VERSION"))
    return nothing
end

struct BackendUnsupported <: Exception
    reason::String
end
Base.showerror(io::IO, e::BackendUnsupported) = print(io, e.reason)

struct ResourceLimitError <: Exception
    message::String
end
Base.showerror(io::IO, e::ResourceLimitError) = print(io, e.message)

function _exact_rational(x)
    x isa Bool && throw(ArgumentError("Bool is not an exact numeric input"))
    if x isa Integer
        return BigInt(x) // BigInt(1)
    elseif x isa Rational
        return BigInt(numerator(x)) // BigInt(denominator(x))
    elseif x isa Union{Float16,Float32,Float64,BigFloat}
        isfinite(x) || throw(ArgumentError("numeric inputs must be finite"))
        x isa BigFloat && return _exact_bigfloat_rational(x)
        return Rational{BigInt}(x)
    end
    throw(ArgumentError("unsupported numeric input type $(typeof(x))"))
end

function _exact_bigfloat_rational(x::BigFloat)
    isfinite(x) || throw(ArgumentError("numeric inputs must be finite"))
    iszero(x) && return BigInt(0) // BigInt(1)

    # MPFR stores a significand and a separate binary exponent. Check both
    # components before expanding that representation into a BigInt. The
    # minimum precision removes significand trailing zeros, so large-precision
    # values such as 1 + 2^-500 remain cheap and exact.
    binary_exponent = exponent(x)
    binary_exponent < _RATIONAL_COMPONENT_BITS ||
        throw(ResourceLimitError("RESOURCE_LIMIT: rational component limit exceeded"))
    binary_exponent < -_MAX_DYADIC_DENOMINATOR_EXPONENT &&
        throw(ResourceLimitError("RESOURCE_LIMIT: rational component limit exceeded"))
    significand_bits = ccall((:mpfr_min_prec, Base.MPFR.libmpfr), Clong,
        (Ref{BigFloat},), x)
    significand_bits <= _RATIONAL_COMPONENT_BITS ||
        throw(ResourceLimitError("RESOURCE_LIMIT: rational component limit exceeded"))
    binary_shift = binary_exponent - (Int(significand_bits) - 1)
    binary_shift < 0 && -binary_shift > _MAX_DYADIC_DENOMINATOR_EXPONENT &&
        throw(ResourceLimitError("RESOURCE_LIMIT: rational component limit exceeded"))
    binary_shift >= 0 && Int(significand_bits) + binary_shift > _RATIONAL_COMPONENT_BITS &&
        throw(ResourceLimitError("RESOURCE_LIMIT: rational component limit exceeded"))
    if binary_shift >= 0
        cmp = ccall((:mpfr_cmp_z, Base.MPFR.libmpfr), Cint,
            (Ref{BigFloat}, Ref{BigInt}), abs(x), _MAX_RATIONAL_COMPONENT_VALUE)
        cmp <= 0 || throw(ResourceLimitError("RESOURCE_LIMIT: rational component limit exceeded"))
    end

    significand_float = setprecision(BigFloat, Int(significand_bits)) do
        ldexp(abs(x), Int(significand_bits) - 1 - binary_exponent)
    end
    cmp_significand = ccall((:mpfr_cmp_z, Base.MPFR.libmpfr), Cint,
        (Ref{BigFloat}, Ref{BigInt}), significand_float, _MAX_RATIONAL_COMPONENT_VALUE)
    cmp_significand <= 0 ||
        throw(ResourceLimitError("RESOURCE_LIMIT: rational component limit exceeded"))
    significand = BigInt(significand_float)
    numerator_value = binary_shift >= 0 ? significand << binary_shift : significand
    denominator_value = binary_shift < 0 ? big(1) << -binary_shift : big(1)
    (ndigits(numerator_value; base=10) <= _RATIONAL_COMPONENT_DIGITS &&
     ndigits(denominator_value; base=10) <= _RATIONAL_COMPONENT_DIGITS) ||
        throw(ResourceLimitError("RESOURCE_LIMIT: rational component limit exceeded"))
    return (signbit(x) ? -numerator_value : numerator_value) // denominator_value
end

function _rational_interval(x::Rational{BigInt}, bits::Int)
    n, d = numerator(x), denominator(x)
    d > 0 || throw(ArgumentError("rational denominator must be positive"))
    nl = BigFloat(n, _RD; precision=bits)
    nu = BigFloat(n, _RU; precision=bits)
    dl = BigFloat(d, _RD; precision=bits)
    du = BigFloat(d, _RU; precision=bits)
    lows = (_div(nl, dl, _RD), _div(nl, du, _RD),
            _div(nu, dl, _RD), _div(nu, du, _RD))
    highs = (_div(nl, dl, _RU), _div(nl, du, _RU),
             _div(nu, dl, _RU), _div(nu, du, _RU))
    return _iv(minimum(lows), maximum(highs), bits)
end

function _iv(lo::BigFloat, hi::BigFloat, bits::Int)
    isfinite(lo) && isfinite(hi) ||
        throw(ProofArithmeticError("proof interval endpoints are not finite"))
    lo <= hi || throw(ArgumentError("proof interval endpoints must be ordered"))
    # The audited IA constructor retains these same-precision endpoints and
    # sets guaranteed/com metadata. The arithmetic establishing them is local.
    x = IA.interval(BigFloat, lo, hi, IA.com)
    (IA.isguaranteed(x) && IA.decoration(x) == IA.com) ||
        throw(ProofArithmeticError("IntervalArithmetic returned unusable metadata"))
    return x
end

_iv(x::Rational{BigInt}, bits::Int) = _rational_interval(x, bits)
_iv(x::Rational, bits::Int) = _rational_interval(BigInt(numerator(x)) // BigInt(denominator(x)), bits)
_iv(x::Integer, bits::Int) = _iv(BigInt(x) // BigInt(1), bits)
function _iv(lo::Rational, hi::Rational, bits::Int)
    l = _iv(lo, bits); h = _iv(hi, bits)
    _iv(_lo(l), _hi(h), bits)
end
_lo(x::IA.Interval{BigFloat}) = IA.inf(x)
_hi(x::IA.Interval{BigFloat}) = IA.sup(x)
_point(x::BigFloat, bits::Int) = _iv(x, x, bits)

function _iadd(x::IA.Interval{BigFloat}, y::IA.Interval{BigFloat}, bits::Int)
    _iv(_add(_lo(x), _lo(y), _RD), _add(_hi(x), _hi(y), _RU), bits)
end

function _isub(x::IA.Interval{BigFloat}, y::IA.Interval{BigFloat}, bits::Int)
    _iv(_sub(_lo(x), _hi(y), _RD), _sub(_hi(x), _lo(y), _RU), bits)
end

function _imul(x::IA.Interval{BigFloat}, y::IA.Interval{BigFloat}, bits::Int)
    xl, xu, yl, yu = _lo(x), _hi(x), _lo(y), _hi(y)
    low = min(_mul(xl, yl, _RD), _mul(xl, yu, _RD),
              _mul(xu, yl, _RD), _mul(xu, yu, _RD))
    high = max(_mul(xl, yl, _RU), _mul(xl, yu, _RU),
               _mul(xu, yl, _RU), _mul(xu, yu, _RU))
    _iv(low, high, bits)
end

function _idiv(x::IA.Interval{BigFloat}, y::IA.Interval{BigFloat}, bits::Int)
    yl, yu = _lo(y), _hi(y)
    (yl <= 0 <= yu) && throw(DivideError())
    xl, xu = _lo(x), _hi(x)
    low = min(_div(xl, yl, _RD), _div(xl, yu, _RD),
              _div(xu, yl, _RD), _div(xu, yu, _RD))
    high = max(_div(xl, yl, _RU), _div(xl, yu, _RU),
               _div(xu, yl, _RU), _div(xu, yu, _RU))
    _iv(low, high, bits)
end

function _isquare(x::IA.Interval{BigFloat}, bits::Int)
    lo, hi = _lo(x), _hi(x)
    low = if lo <= 0 <= hi
        BigFloat(0; precision=bits)
    else
        min(_mul(lo, lo, _RD), _mul(hi, hi, _RD))
    end
    high = max(_mul(lo, lo, _RU), _mul(hi, hi, _RU))
    _iv(low, high, bits)
end

function _iabs(x::IA.Interval{BigFloat}, bits::Int)
    lo, hi = _lo(x), _hi(x)
    lo >= 0 && return x
    hi <= 0 && return _iv(-hi, -lo, bits)
    _iv(BigFloat(0; precision=bits), max(-lo, hi), bits)
end

function _iexp(x::IA.Interval{BigFloat}, bits::Int)
    _iv(_exp(_lo(x), _RD), _exp(_hi(x), _RU), bits)
end

function _ilog(x::IA.Interval{BigFloat}, bits::Int)
    _lo(x) > 0 || throw(DomainError(x, "log interval must be strictly positive"))
    _iv(_log(_lo(x), _RD), _log(_hi(x), _RU), bits)
end

function _pi_interval(bits::Int)
    _iv(BigFloat(π, _RD; precision=bits), BigFloat(π, _RU; precision=bits), bits)
end

_twopi(bits::Int) = _imul(_iv(2, bits), _pi_interval(bits), bits)

function _contains_integer(x::IA.Interval{BigFloat})
    ceil(BigInt, _lo(x)) <= floor(BigInt, _hi(x))
end

function _itrig(x::IA.Interval{BigFloat}, bits::Int, fn::Symbol)
    twopi = _twopi(bits)
    width_lower = _sub(_hi(x), _lo(x), _RD)
    if width_lower >= _hi(twopi)
        return _iv(BigFloat(-1; precision=bits), BigFloat(1; precision=bits), bits)
    end
    lo, hi = _lo(x), _hi(x)
    f = fn === :sin ? _sin : _cos
    lows = (f(lo, _RD), f(hi, _RD))
    highs = (f(lo, _RU), f(hi, _RU))
    lower, upper = minimum(lows), maximum(highs)
    if fn === :sin
        jmax = _isub(_idiv(x, twopi, bits), _iv(1//4, bits), bits)
        jmin = _iadd(_idiv(x, twopi, bits), _iv(1//4, bits), bits)
        _contains_integer(jmax) && (upper = BigFloat(1; precision=bits))
        _contains_integer(jmin) && (lower = BigFloat(-1; precision=bits))
    else
        jmax = _idiv(x, twopi, bits)
        jmin = _isub(_idiv(x, twopi, bits), _iv(1//2, bits), bits)
        _contains_integer(jmax) && (upper = BigFloat(1; precision=bits))
        _contains_integer(jmin) && (lower = BigFloat(-1; precision=bits))
    end
    return _iv(lower, upper, bits)
end

_isin(x::IA.Interval{BigFloat}, bits::Int) = _itrig(x, bits, :sin)
_icos(x::IA.Interval{BigFloat}, bits::Int) = _itrig(x, bits, :cos)

function _interval_matrix(A::AbstractMatrix{<:Rational}, bits::Int)
    out = Matrix{IA.Interval{BigFloat}}(undef, size(A))
    for j in axes(A, 2), i in axes(A, 1)
        out[i, j] = _iv(A[i, j], bits)
    end
    out
end

function _effective_exact_matrix(A, label::String)
    A isa AbstractMatrix || throw(ArgumentError("$label must be a matrix"))
    Base.require_one_based_indexing(A)
    m = Matrix{Rational{BigInt}}(undef, size(A))
    for j in axes(A, 2), i in axes(A, 1)
        m[i, j] = _exact_rational(A[i, j])
    end
    return m
end

struct CertificationModel
    Q::Matrix{Rational{BigInt}}
    L::Matrix{Rational{BigInt}}
    K::Matrix{Rational{BigInt}}
    phases::Vector{Rational{BigInt}}
    s::Union{Nothing,Rational{BigInt}}
    resource_reason::Union{Nothing,String}
    n::Int
    M::Int
end

CertificationModel(Q::Matrix{Rational{BigInt}}, L::Matrix{Rational{BigInt}},
        K::Matrix{Rational{BigInt}}, phases::Vector{Rational{BigInt}},
        s::Rational{BigInt}) = CertificationModel(Q, L, K, phases, s, nothing,
            size(Q, 1), size(Q, 2))

const _CertificationInterpretation = NamedTuple{
    (:basis, :orientation, :theta_period, :phase_units, :coefficient_convention,
     :numeric_semantics, :spectrum_convention, :physical_normalization, :s),
    Tuple{String, String, String, String, String, String, String, String, Union{Nothing,Rational{BigInt}}}}

"""Exact input identity and interpretation carried by every public result."""
struct CertificationIdentity
    input_sha256::Union{Nothing,String}
    input_identity_status::Symbol
    input_identity_reason::Union{Nothing,String}
    interpretation::_CertificationInterpretation
end

function _validate_scalar_input(x, label::String; integer_only::Bool=false)
    x isa Bool && throw(ArgumentError("Bool is not an exact numeric input"))
    if integer_only
        is_integer_value = x isa Integer ||
            (x isa Rational && denominator(x) == 1) ||
            (x isa AbstractFloat && isfinite(x) && isinteger(x))
        is_integer_value || throw(ArgumentError("$label entries must be integers"))
    elseif !(x isa Union{Integer,Rational,Float16,Float32,Float64,BigFloat})
        throw(ArgumentError("$label contains an unsupported numeric input"))
    end
    x isa AbstractFloat && !isfinite(x) && throw(ArgumentError("$label entries must be finite"))
    nothing
end

function _resource_limited_model(n::Int, M::Int)
    empty_matrix = Matrix{Rational{BigInt}}(undef, 0, 0)
    CertificationModel(empty_matrix, empty_matrix, empty_matrix,
        Rational{BigInt}[], nothing, "RESOURCE_LIMIT", n, M)
end

function prepare_model(Q, L, K; phases=nothing)
    Q isa AbstractMatrix || throw(ArgumentError("Q must be a matrix"))
    Base.require_one_based_indexing(Q)
    n, M = size(Q)
    n >= 1 || throw(DimensionMismatch("Q must have at least one row"))
    M >= 1 || throw(DimensionMismatch("Q must have at least one column"))
    L isa AbstractMatrix || throw(ArgumentError("L must be a matrix"))
    K isa AbstractMatrix || throw(ArgumentError("K must be a matrix"))
    Base.require_one_based_indexing(L)
    Base.require_one_based_indexing(K)
    size(L) == (2, M) || throw(DimensionMismatch("L must have size (2,M)"))
    size(K) == (n, n) || throw(DimensionMismatch("K must have size (n,n)"))
    if phases !== nothing
        phases isa AbstractVector || throw(ArgumentError("phases must be a vector"))
        Base.require_one_based_indexing(phases)
        length(phases) == M || throw(DimensionMismatch("phases must have length M"))
    end

    # Validate shape/type/finiteness/symmetry before applying resource limits,
    # so malformed inputs retain their ordinary argument errors.
    for value in Q
        _validate_scalar_input(value, "Q"; integer_only=true)
    end
    for value in L
        _validate_scalar_input(value, "L")
    end
    for value in K
        _validate_scalar_input(value, "K")
    end
    for j in 1:n, i in 1:j-1
        K[i, j] == K[j, i] || throw(ArgumentError("K must be exactly symmetric"))
    end
    if phases !== nothing
        for value in phases
            _validate_scalar_input(value, "phases")
        end
    end

    # Do not duplicate arrays outside the schema's bounded dimensions. The
    # typed certification operations return UNSUPPORTED for this model.
    (n <= 32 && M <= 2048) || return _resource_limited_model(n, M)

    try
        q = _effective_exact_matrix(Q, "Q")
        all(denominator(x) == 1 for x in q) || throw(ArgumentError("Q entries must be integers"))
        l = _effective_exact_matrix(L, "L")
        k = _effective_exact_matrix(K, "K")
        p = phases === nothing ? fill(BigInt(0)//BigInt(1), M) :
            [_exact_rational(x) for x in phases]
        s = maximum(view(l, 2, :))
        return CertificationModel(q, l, k, p, s)
    catch error
        error isa ResourceLimitError || rethrow()
        return _resource_limited_model(n, M)
    end
end

function _exact_seed(seed, n::Int)
    seed isa AbstractVector || throw(ArgumentError("theta0 must be a vector"))
    Base.require_one_based_indexing(seed)
    length(seed) == n || throw(DimensionMismatch("theta0 must have length n"))
    [_exact_rational(x) for x in seed]
end

function _amplitudes(model::CertificationModel, bits::Int)
    log10 = _ilog(_iv(10, bits), bits)
    out = Vector{IA.Interval{BigFloat}}(undef, size(model.Q, 2))
    for a in eachindex(out)
        coeff = model.L[1, a]
        delta = model.L[2, a] - model.s
        if iszero(coeff)
            out[a] = _iv(0, bits)
        else
            power = _iexp(_imul(_iv(delta, bits), log10, bits), bits)
            out[a] = _imul(_iv(coeff, bits), power, bits)
        end
    end
    out
end

function _phase_intervals(model::CertificationModel, bits::Int)
    tau = _twopi(bits)
    ph = Vector{IA.Interval{BigFloat}}(undef, length(model.phases))
    for a in eachindex(ph)
        ph[a] = _iv(model.phases[a], bits)
    end
    return tau, ph
end

function _phase_value(model::CertificationModel, theta::Vector{<:IA.Interval{BigFloat}}, a::Int, bits::Int)
    tau, phases = _phase_intervals(model, bits)
    value = _iv(0, bits)
    for i in eachindex(theta)
        charge = BigInt(numerator(model.Q[i, a]))
        iszero(charge) && continue
        term = _imul(_iv(charge, bits), theta[i], bits)
        value = _iadd(value, term, bits)
    end
    _iadd(_imul(tau, value, bits), phases[a], bits)
end

function _gradient_hessian(model::CertificationModel, theta::Vector{<:IA.Interval{BigFloat}}, bits::Int)
    n, M = size(model.Q)
    amps = _amplitudes(model, bits)
    tau, phases = _phase_intervals(model, bits)
    grad = [_iv(0, bits) for _ in 1:n]
    H = Matrix{IA.Interval{BigFloat}}(undef, n, n)
    for j in 1:n, i in 1:n
        H[i, j] = _iv(0, bits)
    end
    for a in 1:M
        iszero(model.L[1, a]) && continue
        phase = _iv(0, bits)
        for i in 1:n
            charge = BigInt(numerator(model.Q[i, a]))
            iszero(charge) && continue
            phase = _iadd(phase, _imul(_iv(charge, bits), theta[i], bits), bits)
        end
        phase = _iadd(_imul(tau, phase, bits), phases[a], bits)
        sn, cs = _isin(phase, bits), _icos(phase, bits)
        for i in 1:n
            qi = BigInt(numerator(model.Q[i, a]))
            iszero(qi) && continue
            coeff = _imul(_imul(tau, amps[a], bits), sn, bits)
            grad[i] = _iadd(grad[i], _imul(_iv(qi, bits), coeff, bits), bits)
        end
        # Build the upper triangle once; mirror identical interval objects.
        for j in 1:n, i in 1:j
            qi = BigInt(numerator(model.Q[i, a]))
            qj = BigInt(numerator(model.Q[j, a]))
            (iszero(qi) || iszero(qj)) && continue
            coeff = _imul(_imul(_imul(tau, tau, bits), amps[a], bits), cs, bits)
            entry = _imul(_iv(qi*qj, bits), coeff, bits)
            H[i, j] = _iadd(H[i, j], entry, bits)
        end
    end
    for j in 1:n, i in 1:j-1
        H[j, i] = H[i, j]
    end
    return grad, H
end

_gradient(model::CertificationModel, theta::Vector{<:IA.Interval{BigFloat}}, bits::Int) =
    first(_gradient_hessian(model, theta, bits))
_hessian(model::CertificationModel, theta::Vector{<:IA.Interval{BigFloat}}, bits::Int) =
    last(_gradient_hessian(model, theta, bits))

function _point_intervals(x::Vector{Rational{BigInt}}, bits::Int)
    [_iv(v, bits) for v in x]
end

function _isvalidproof(x::IA.Interval{BigFloat})
    isfinite(_lo(x)) && isfinite(_hi(x)) && _lo(x) <= _hi(x) &&
        IA.isguaranteed(x) && IA.decoration(x) == IA.com
end
