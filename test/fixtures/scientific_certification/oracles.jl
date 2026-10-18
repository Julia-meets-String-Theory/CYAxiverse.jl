"""Independent analytic and exact-algebra reference calculations."""

using LinearAlgebra

const _ORACLE_RAT = Rational{BigInt}
const _ORACLE_D = RoundDown
const _ORACLE_U = RoundUp

_orat(x::Integer) = BigInt(x) // BigInt(1)
_orat(x::Rational) = BigInt(numerator(x)) // BigInt(denominator(x))
_oracle_exact(x::BigFloat, bits::Int) = setprecision(BigFloat, bits) do; Rational{BigInt}(x); end

function _oracle_float(x::_ORACLE_RAT, bits::Int, rounding::RoundingMode)
    BigFloat(x, rounding; precision=bits)
end

function _oracle_directed(f::Function, bits::Int, rounding::RoundingMode)
    setprecision(BigFloat, bits) do
        setrounding(BigFloat, rounding) do
            f()
        end
    end
end

function oracle_pi_interval(bits=512)
    lo = BigFloat(pi, _ORACLE_D; precision=bits)
    hi = BigFloat(pi, _ORACLE_U; precision=bits)
    _oracle_exact(lo, bits), _oracle_exact(hi, bits)
end

function oracle_c_interval(bits=512)
    plo, phi = oracle_pi_interval(bits)
    lo = _oracle_directed(bits, _ORACLE_D) do
        (BigFloat(2) * BigFloat(plo, _ORACLE_D; precision=bits))^2
    end
    hi = _oracle_directed(bits, _ORACLE_U) do
        (BigFloat(2) * BigFloat(phi, _ORACLE_U; precision=bits))^2
    end
    _oracle_exact(lo, bits), _oracle_exact(hi, bits)
end

function oracle_scalar_root(phi::Rational)
    # Independent analytic root for q=1 and an exact radian phase.
    p_lo_pi, p_hi_pi = oracle_pi_interval(512)
    phase = _orat(phi)
    p_lo = _oracle_float(phase, 512, _ORACLE_D)
    p_hi = _oracle_float(phase, 512, _ORACLE_U)
    if phase >= 0
        lo = _oracle_directed(512, _ORACLE_D) do
            -p_hi / (BigFloat(2) * BigFloat(p_lo_pi, _ORACLE_D; precision=512))
        end
        hi = _oracle_directed(512, _ORACLE_U) do
            -p_lo / (BigFloat(2) * BigFloat(p_hi_pi, _ORACLE_U; precision=512))
        end
    else
        lo = _oracle_directed(512, _ORACLE_D) do
            -p_hi / (BigFloat(2) * BigFloat(p_hi_pi, _ORACLE_U; precision=512))
        end
        hi = _oracle_directed(512, _ORACLE_U) do
            -p_lo / (BigFloat(2) * BigFloat(p_lo_pi, _ORACLE_D; precision=512))
        end
    end
    _oracle_exact(lo, 512), _oracle_exact(hi, 512)
end

function oracle_scalar_curvature(log10_coefficient::Rational; bits=512)
    # Exact integer powers and an independent directed sqrt(10) enclosure.
    c_lo, c_hi = oracle_c_interval(bits)
    x = _orat(log10_coefficient)
    if denominator(x) == 1
        exponent = numerator(x)
        abs(exponent) <= 1000 || throw(ArgumentError("integer reference exponent is outside the bounded oracle"))
        scale = exponent >= 0 ? BigInt(10)^Int(exponent)//BigInt(1) :
            BigInt(1)//BigInt(10)^Int(-exponent)
        lo = _oracle_directed(bits, _ORACLE_D) do
            BigFloat(c_lo, _ORACLE_D; precision=bits) * BigFloat(scale, _ORACLE_D; precision=bits)
        end
        hi = _oracle_directed(bits, _ORACLE_U) do
            BigFloat(c_hi, _ORACLE_U; precision=bits) * BigFloat(scale, _ORACLE_U; precision=bits)
        end
    elseif x == -1//2
        sqrt_lo = _oracle_directed(bits, _ORACLE_D) do
            sqrt(BigFloat(10))
        end
        sqrt_hi = _oracle_directed(bits, _ORACLE_U) do
            sqrt(BigFloat(10))
        end
        lo = _oracle_directed(bits, _ORACLE_D) do
            BigFloat(c_lo, _ORACLE_D; precision=bits) / sqrt_hi
        end
        hi = _oracle_directed(bits, _ORACLE_U) do
            BigFloat(c_hi, _ORACLE_U; precision=bits) / sqrt_lo
        end
    else
        throw(ArgumentError("independent curvature oracle supports integer powers and -1/2"))
    end
    _oracle_exact(lo, bits), _oracle_exact(hi, bits)
end

function oracle_integer_quadratic(Q::Matrix{<:Integer}, weights::Vector{<:Integer})
    size(Q,2) == length(weights) || throw(DimensionMismatch())
    Q * Diagonal(weights) * transpose(Q)
end

function oracle_t04_intervals(bits=512)
    # Exact pencil has roots +/-1/sqrt(3); enclose using directed MPFR.
    root_lo = _oracle_directed(bits, _ORACLE_D) do
        sqrt(BigFloat(3))
    end
    root_hi = _oracle_directed(bits, _ORACLE_U) do
        sqrt(BigFloat(3))
    end
    c_lo, c_hi = oracle_c_interval(bits)
    pos_lo = _oracle_directed(bits, _ORACLE_D) do
        BigFloat(c_lo, _ORACLE_D; precision=bits) / root_hi
    end
    pos_hi = _oracle_directed(bits, _ORACLE_U) do
        BigFloat(c_hi, _ORACLE_U; precision=bits) / root_lo
    end
    neg_lo, neg_hi = -_oracle_exact(pos_hi,bits), -_oracle_exact(pos_lo,bits)
    ((neg_lo, neg_hi), (_oracle_exact(pos_lo,bits), _oracle_exact(pos_hi,bits)))
end

function oracle_t14_intervals(bits=512)
    # Characteristic polynomial uses the exact binary Float64 values 0.1, 0.3.
    r = Rational{BigInt}(Float64(0.1))
    k = Rational{BigInt}(Float64(0.3))
    a = 2 - k^2
    b = 1 - 2r
    d = -r
    disc = b^2 - 4a*d
    a > 0 && disc > 0 || error("T14 exact quadratic is not in its expected domain")
    dlo = _oracle_directed(bits, _ORACLE_D) do
        sqrt(BigFloat(disc, _ORACLE_D; precision=bits))
    end
    dhi = _oracle_directed(bits, _ORACLE_U) do
        sqrt(BigFloat(disc, _ORACLE_U; precision=bits))
    end
    alo = _oracle_float(a, bits, _ORACLE_D)
    ahi = _oracle_float(a, bits, _ORACLE_U)
    blo = _oracle_float(b, bits, _ORACLE_D)
    bhi = _oracle_float(b, bits, _ORACLE_U)
    denlo = _oracle_directed(bits, _ORACLE_D) do
        2 * alo
    end
    denhi = _oracle_directed(bits, _ORACLE_U) do
        2 * ahi
    end
    neg_lo = _oracle_directed(bits, _ORACLE_D) do
        (-bhi - dhi) / denlo
    end
    neg_hi = _oracle_directed(bits, _ORACLE_U) do
        (-blo - dlo) / denhi
    end
    pos_lo = _oracle_directed(bits, _ORACLE_D) do
        (-bhi + dlo) / denhi
    end
    pos_hi = _oracle_directed(bits, _ORACLE_U) do
        (-blo + dhi) / denlo
    end
    c_lo, c_hi = oracle_c_interval(bits)
    nlo = _oracle_directed(bits, _ORACLE_D) do
        BigFloat(neg_lo) * BigFloat(c_hi, _ORACLE_U; precision=bits)
    end
    nhi = _oracle_directed(bits, _ORACLE_U) do
        BigFloat(neg_hi) * BigFloat(c_lo, _ORACLE_D; precision=bits)
    end
    plo = _oracle_directed(bits, _ORACLE_D) do
        BigFloat(pos_lo) * BigFloat(c_lo, _ORACLE_D; precision=bits)
    end
    phi = _oracle_directed(bits, _ORACLE_U) do
        BigFloat(pos_hi) * BigFloat(c_hi, _ORACLE_U; precision=bits)
    end
    ((_oracle_exact(nlo,bits), _oracle_exact(nhi,bits)),
     (_oracle_exact(plo,bits), _oracle_exact(phi,bits)))
end

function _poly_trim(p::Vector{_ORACLE_RAT})
    while length(p) > 1 && iszero(p[end])
        pop!(p)
    end
    p
end

function _poly_derivative(p::Vector{_ORACLE_RAT})
    length(p) <= 1 && return _ORACLE_RAT[0//1]
    _ORACLE_RAT[BigInt(i-1) * p[i] for i in 2:length(p)]
end

function _poly_remainder(a0::Vector{_ORACLE_RAT}, b0::Vector{_ORACLE_RAT})
    a = copy(a0); b = copy(b0)
    iszero(b[end]) && error("zero polynomial divisor")
    while length(a) >= length(b) && any(!iszero, a)
        shift = length(a) - length(b)
        factor = a[end] / b[end]
        for j in eachindex(b)
            a[shift+j] -= factor * b[j]
        end
        _poly_trim(a)
    end
    _poly_trim(a)
end

function _poly_eval(p::Vector{_ORACLE_RAT}, x::_ORACLE_RAT)
    value = _ORACLE_RAT(0)
    for coefficient in Iterators.reverse(p)
        value = value*x + coefficient
    end
    value
end

function _sturm_sequence(p::Vector{_ORACLE_RAT})
    seq = [_poly_trim(copy(p)), _poly_derivative(p)]
    while any(!iszero, seq[end])
        rem = _poly_remainder(seq[end-1], seq[end])
        all(iszero, rem) && break
        push!(seq, -rem)
    end
    seq
end

function _variations(seq, x::_ORACLE_RAT)
    signs = Int[]
    for p in seq
        v = _poly_eval(p, x)
        iszero(v) && continue
        push!(signs, v > 0 ? 1 : -1)
    end
    count(i -> signs[i] != signs[i+1], 1:max(0, length(signs)-1))
end

function _sturm_count(seq, lo::_ORACLE_RAT, hi::_ORACLE_RAT)
    _variations(seq, lo) - _variations(seq, hi)
end

function oracle_t05_reference(index::Int, claim::Tuple{_ORACLE_RAT,_ORACLE_RAT}, poly::Vector{_ORACLE_RAT})
    starts = (4297459, 9349281, 15337845, 23207141, 42599543)
    1 <= index <= length(starts) || throw(BoundsError(starts, index))
    # det(A-mu*K), independently evaluated with exact rational Sturm arithmetic.
    seq = _sturm_sequence(poly)
    lo = BigInt(starts[index]) // BigInt(2)^24
    hi = BigInt(starts[index]+1) // BigInt(2)^24
    _sturm_count(seq, lo, hi) == 1 || error("published T05 isolating interval lost its unique root")
    for _ in 1:8192
        claim[1] <= lo && hi <= claim[2] && return (lo, hi)
        mid = (lo+hi)/2
        if iszero(_poly_eval(poly, mid))
            return (mid, mid)
        elseif _sturm_count(seq, lo, mid) == 1
            hi = mid
        else
            lo = mid
        end
    end
    error("T05 exact oracle could not be refined inside the certified claim")
end

function oracle_t05_scaled_reference(index::Int, claim::Tuple{_ORACLE_RAT,_ORACLE_RAT}, poly::Vector{_ORACLE_RAT})
    starts = (4297459, 9349281, 15337845, 23207141, 42599543)
    1 <= index <= length(starts) || throw(BoundsError(starts, index))
    seq = _sturm_sequence(poly)
    lo = BigInt(starts[index]) // BigInt(2)^24
    hi = BigInt(starts[index]+1) // BigInt(2)^24
    _sturm_count(seq, lo, hi) == 1 || error("published T05 isolating interval lost its unique root")
    c_lo,c_hi=oracle_c_interval(512)
    for _ in 1:8192
        scaled=(c_lo*lo,c_hi*hi)
        claim[1] <= scaled[1] && scaled[2] <= claim[2] && return scaled
        mid=(lo+hi)/2
        if iszero(_poly_eval(poly,mid))
            lo=mid; hi=mid
        elseif _sturm_count(seq,lo,mid)==1
            hi=mid
        else
            lo=mid
        end
    end
    error("T05 exact scaled oracle could not be refined inside the certified claim")
end

function oracle_contains_interval(claim::Tuple, reference::Tuple)
    claim[1] <= reference[1] && reference[2] <= claim[2]
end

function oracle_close(a, b; rtol=big"1e-60")
    abs(BigFloat(a) - BigFloat(b)) <= rtol * max(abs(BigFloat(a)), abs(BigFloat(b)), BigFloat(1))
end

function _poly_add(a::Vector{_ORACLE_RAT}, b::Vector{_ORACLE_RAT})
    out = fill(_ORACLE_RAT(0), max(length(a), length(b)))
    for i in eachindex(a); out[i] += a[i]; end
    for i in eachindex(b); out[i] += b[i]; end
    _poly_trim(out)
end

function _poly_multiply(a::Vector{_ORACLE_RAT}, b::Vector{_ORACLE_RAT})
    out = fill(_ORACLE_RAT(0), length(a)+length(b)-1)
    for i in eachindex(a),j in eachindex(b)
        out[i+j-1] += a[i]*b[j]
    end
    _poly_trim(out)
end

function _permutations(xs::Vector{Int})
    length(xs) <= 1 && return [copy(xs)]
    out=Vector{Vector{Int}}()
    for i in eachindex(xs)
        rest=[xs[j] for j in eachindex(xs) if j!=i]
        for tail in _permutations(rest)
            push!(out,[xs[i];tail])
        end
    end
    out
end

function _permutation_sign(p::Vector{Int})
    inversions=0
    for i in eachindex(p),j in i+1:length(p)
        p[i]>p[j] && (inversions+=1)
    end
    iseven(inversions) ? 1 : -1
end

function oracle_exact_pencil_polynomial(A::Matrix{_ORACLE_RAT},K::Matrix{_ORACLE_RAT})
    n,m=size(A)
    n==m && size(K)==(n,n) || throw(DimensionMismatch())
    1<=n<=5 || throw(ArgumentError("the exact polynomial oracle is bounded to n=5"))
    total=fill(_ORACLE_RAT(0),n+1)
    for permutation in _permutations(collect(1:n))
        term=_ORACLE_RAT[1//1]
        for i in 1:n
            factor=_ORACLE_RAT[A[i,permutation[i]],-K[i,permutation[i]]]
            term=_poly_multiply(term,factor)
        end
        _poly_add!(total,_permutation_sign(permutation).*term)
    end
    _poly_trim(total)
end

function oracle_t05_pencil_polynomial(Q::Matrix{_ORACLE_RAT},L::Matrix{_ORACLE_RAT},
        K::Matrix{_ORACLE_RAT})
    size(Q)==(5,9) && size(L)==(2,9) && size(K)==(5,5) ||
        throw(DimensionMismatch("T05 oracle requires its fixed 5-by-9 model data"))
    all(denominator(x)==1 for x in Q) && all(denominator(x)==1 for x in L[1,:]) &&
        all(denominator(x)==1 for x in K) || throw(ArgumentError("T05 oracle requires exact integer model data"))
    all(==(1//1),L[1,:]) && all(iszero,L[2,:]) ||
        throw(ArgumentError("T05 oracle coefficients/logs differ from the fixed family"))
    A=Matrix{_ORACLE_RAT}(Q*Diagonal(L[1,:])*transpose(Q))
    polynomial=oracle_exact_pencil_polynomial(A,K)
    expected=_ORACLE_RAT[55//1,-435//1,1161//1,-1341//1,678//1,-120//1]
    polynomial==expected || error("independent T05 pencil reproduction differs from the fixed exact coefficients")
    polynomial
end

function _poly_add!(a::Vector{_ORACLE_RAT}, b::Vector{_ORACLE_RAT})
    resize!(a,max(length(a),length(b)))
    for i in 1:length(b); a[i]+=b[i]; end
    _poly_trim(a)
end

function oracle_refine_scaled_root(poly::Vector{_ORACLE_RAT}, isolator::Tuple{_ORACLE_RAT,_ORACLE_RAT},
        claim::Tuple{_ORACLE_RAT,_ORACLE_RAT}, bits=512)
    seq=_sturm_sequence(poly)
    lo,hi=isolator
    lo<hi && _sturm_count(seq,lo,hi)==1 || error("oracle isolator must contain exactly one root")
    c_lo,c_hi=oracle_c_interval(bits)
    for _ in 1:8192
        ref=(c_lo*lo,c_hi*hi)
        claim[1] <= ref[1] && ref[2] <= claim[2] && return ref
        mid=(lo+hi)/2
        if iszero(_poly_eval(poly,mid))
            lo=mid; hi=mid
        elseif _sturm_count(seq,lo,mid)==1
            hi=mid
        else
            lo=mid
        end
    end
    error("independent held-out oracle could not be refined inside the submitted claim")
end
