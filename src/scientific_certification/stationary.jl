struct ProofOptions
    precisions::Tuple{Vararg{Int}}
    max_newton_steps::Int
    max_backtracks::Int
    box_radii::Tuple{Vararg{Rational{BigInt}}}
    max_expansions::Int
    max_inertia_evaluations::Int
    relative_width::Rational{BigInt}
    spectral_goal::Symbol
    max_dimension::Int
    max_terms::Int
end

function _option_int(name::String, value, lower::Int, upper::Int)
    (value isa Integer && !(value isa Bool)) || throw(ArgumentError("$name must be an integer"))
    lower <= value <= upper || throw(ArgumentError("$name must be in $lower:$upper"))
    Int(value)
end

function ProofOptions(; precisions=(256, 512), max_newton_steps=20,
        max_backtracks=8, box_radii=(big(1)//big(2)^20, big(1)//big(2)^30,
        big(1)//big(2)^40, big(1)//big(2)^60), max_expansions=32,
        max_inertia_evaluations=4096, relative_width=big(1)//big(1048576),
        spectral_goal=:clusters, max_dimension=32, max_terms=2048)
    precisions isa Tuple || precisions isa AbstractVector ||
        throw(ArgumentError("precisions must be a tuple or vector"))
    isempty(precisions) && throw(ArgumentError("precisions cannot be empty"))
    ps = Int[]
    for p in precisions
        _option_int("precision", p, 2, 4096)
        p in (256, 512) || throw(ArgumentError("qualified precisions are 256 and 512"))
        push!(ps, Int(p))
    end
    issorted(ps) && length(unique(ps)) == length(ps) ||
        throw(ArgumentError("precisions must be a strictly increasing subsequence"))
    box_radii isa Tuple || box_radii isa AbstractVector ||
        throw(ArgumentError("box_radii must be a tuple or vector"))
    1 <= length(box_radii) <= 4 || throw(ArgumentError("one to four box radii are supported"))
    rs = Rational{BigInt}[_exact_rational(x) for x in box_radii]
    all(r -> 0 < r < 1//2, rs) || throw(ArgumentError("box radii must be positive and below 1/2"))
    all(rs[i] > rs[i+1] for i in 1:length(rs)-1) ||
        throw(ArgumentError("box radii must be strictly decreasing"))
    rw = _exact_rational(relative_width)
    0 < rw <= big(1)//big(1048576) ||
        throw(ArgumentError("relative_width must be positive and no larger than 2^-20"))
    spectral_goal in (:clusters, :individuals) ||
        throw(ArgumentError("spectral_goal must be :clusters or :individuals"))
    return ProofOptions(Tuple(ps),
        _option_int("max_newton_steps", max_newton_steps, 0, 20),
        _option_int("max_backtracks", max_backtracks, 0, 8), Tuple(rs),
        _option_int("max_expansions", max_expansions, 0, 32),
        _option_int("max_inertia_evaluations", max_inertia_evaluations, 0, 4096),
        rw, spectral_goal,
        _option_int("max_dimension", max_dimension, 1, 32),
        _option_int("max_terms", max_terms, 1, 2048))
end

struct RootResult
    status::Symbol
    model_signature::String
    reason::String
    original_seed::Vector{Rational{BigInt}}
    translation::Vector{BigInt}
    normalized_seed::Vector{Rational{BigInt}}
    center::Vector{Rational{BigInt}}
    box::Vector{IA.Interval{BigFloat}}
    preconditioner::Matrix{Rational{BigInt}}
    F_center::Vector{IA.Interval{BigFloat}}
    H_box::Matrix{IA.Interval{BigFloat}}
    W::Vector{IA.Interval{BigFloat}}
    q_upper::Rational{BigInt}
    precision::Int
    options::ProofOptions
    hessian_classification::Symbol
    hessian_method::Symbol
    hessian_pivots::Vector{IA.Interval{BigFloat}}
    candidate_provenance::Dict{String,String}
    attempts::Vector{String}
end

function _empty_root(status::Symbol, reason::String, seed=Rational{BigInt}[], options=ProofOptions())
    return RootResult(status, "", reason, copy(seed), BigInt[], Rational{BigInt}[],
        Rational{BigInt}[], IA.Interval{BigFloat}[],
        Matrix{Rational{BigInt}}(undef, 0, 0), IA.Interval{BigFloat}[],
        Matrix{IA.Interval{BigFloat}}(undef, 0, 0), IA.Interval{BigFloat}[],
        BigInt(0)//BigInt(1), 0, options, :UNRESOLVED, :none,
        IA.Interval{BigFloat}[], Dict("status"=>"unknown"), String[])
end

function _matrix_product(A::Matrix{IA.Interval{BigFloat}},
                         B::Matrix{IA.Interval{BigFloat}}, bits::Int)
    size(A, 2) == size(B, 1) || throw(DimensionMismatch())
    C = Matrix{IA.Interval{BigFloat}}(undef, size(A, 1), size(B, 2))
    for j in axes(C, 2), i in axes(C, 1)
        value = _iv(0, bits)
        for k in axes(A, 2)
            value = _iadd(value, _imul(A[i, k], B[k, j], bits), bits)
        end
        C[i, j] = value
    end
    C
end

function _matrix_vector(A::Matrix{IA.Interval{BigFloat}},
                        x::Vector{IA.Interval{BigFloat}}, bits::Int)
    size(A, 2) == length(x) || throw(DimensionMismatch())
    out = Vector{IA.Interval{BigFloat}}(undef, size(A, 1))
    for i in axes(A, 1)
        value = _iv(0, bits)
        for j in axes(A, 2)
            value = _iadd(value, _imul(A[i, j], x[j], bits), bits)
        end
        out[i] = value
    end
    out
end

function _interval_ldl(A::Matrix{IA.Interval{BigFloat}}, bits::Int)
    n, m = size(A)
    n == m || throw(DimensionMismatch("LDL input must be square"))
    L = Matrix{IA.Interval{BigFloat}}(undef, n, n)
    for j in 1:n, i in 1:n
        L[i, j] = _iv(i == j ? 1 : 0, bits)
    end
    D = Vector{IA.Interval{BigFloat}}(undef, n)
    signs = Symbol[]
    for j in 1:n
        pivot = A[j, j]
        for k in 1:j-1
            pivot = _isub(pivot, _imul(_isquare(L[j, k], bits), D[k], bits), bits)
        end
        D[j] = pivot
        if _lo(pivot) > 0
            push!(signs, :positive)
        elseif _hi(pivot) < 0
            push!(signs, :negative)
        else
            return (ok=false, pivots=D[1:j], signs=signs, L=L, D=D)
        end
        for i in j+1:n
            value = A[i, j]
            for k in 1:j-1
                value = _isub(value, _imul(_imul(L[i, k], L[j, k], bits), D[k], bits), bits)
            end
            L[i, j] = _idiv(value, pivot, bits)
        end
    end
    return (ok=true, pivots=D, signs=signs, L=L, D=D)
end

function _classify(signs::Vector{Symbol}, n::Int)
    length(signs) == n || return :UNRESOLVED
    positives = count(==(:positive), signs)
    negatives = count(==(:negative), signs)
    positives == n && return :MINIMUM
    negatives == n && return :MAXIMUM
    positives > 0 && negatives > 0 && return :SADDLE
    :UNRESOLVED
end

function _nearest(q::Rational{BigInt}, bits::Int)
    BigFloat(q; precision=bits)
end

function _interval_mid(x::IA.Interval{BigFloat}, bits::Int)
    setrounding(BigFloat, RoundNearest) do
        (_lo(x) + _hi(x)) / BigFloat(2; precision=bits)
    end
end

function _big_solve(Ain::Matrix{BigFloat}, bin::Vector{BigFloat})
    A = copy(Ain); b = copy(bin)
    n = length(b)
    size(A) == (n, n) || throw(DimensionMismatch())
    for k in 1:n
        pivot = k
        magnitude = abs(A[k, k])
        for i in k+1:n
            candidate = abs(A[i, k])
            if candidate > magnitude
                magnitude = candidate; pivot = i
            end
        end
        iszero(magnitude) && throw(SingularException(k))
        if pivot != k
            A[k, :], A[pivot, :] = copy(A[pivot, :]), copy(A[k, :])
            b[k], b[pivot] = b[pivot], b[k]
        end
        for i in k+1:n
            factor = A[i, k] / A[k, k]
            A[i, k] = 0
            for j in k+1:n
                A[i, j] -= factor * A[k, j]
            end
            b[i] -= factor * b[k]
        end
    end
    x = similar(b)
    for i in n:-1:1
        value = b[i]
        for j in i+1:n
            value -= A[i, j] * x[j]
        end
        x[i] = value / A[i, i]
    end
    x
end

function _big_inverse(A::Matrix{BigFloat})
    n = size(A, 1)
    inverse = Matrix{BigFloat}(undef, n, n)
    for j in 1:n
        b = [BigFloat(i == j ? 1 : 0; precision=precision(A[1, 1])) for i in 1:n]
        inverse[:, j] = _big_solve(A, b)
    end
    inverse
end

function _numeric_derivatives(model::CertificationModel, x::Vector{BigFloat}, bits::Int)
    g, H = _gradient_hessian(model, [_point(v, bits) for v in x], bits)
    return BigFloat[_interval_mid(v, bits) for v in g],
           BigFloat[_interval_mid(H[i, j], bits) for i in axes(H, 1), j in axes(H, 2)]
end

function _proposal(model::CertificationModel, seed::Vector{Rational{BigInt}},
                   options::ProofOptions, bits::Int)
    x = BigFloat[_nearest(v, bits) for v in seed]
    for _ in 1:options.max_newton_steps
        g, H = _numeric_derivatives(model, x, bits)
        residual = maximum(abs, g; init=BigFloat(0; precision=bits))
        iszero(residual) && break
        step = try
            _big_solve(H, g)
        catch e
            e isa SingularException && break
            rethrow()
        end
        accepted = false
        for backtrack in 0:options.max_backtracks
            alpha = BigFloat(1; precision=bits) / BigFloat(2^backtrack; precision=bits)
            trial = x .- alpha .* step
            all(isfinite, trial) || continue
            trial_g, _ = _numeric_derivatives(model, trial, bits)
            trial_residual = maximum(abs, trial_g; init=BigFloat(0; precision=bits))
            if isfinite(trial_residual) && trial_residual < residual
                x = trial; accepted = true; break
            end
        end
        accepted || break
    end
    x
end

function _root_witness(model::CertificationModel, original::Vector{Rational{BigInt}},
                       translation::Vector{BigInt}, normalized::Vector{Rational{BigInt}},
                       center::Vector{Rational{BigInt}}, radius::Rational{BigInt},
                       bits::Int, options::ProofOptions, provenance::Dict{String,String}, attempts::Vector{String})
    n = length(center)
    box_bounds = Tuple{Rational{BigInt},Rational{BigInt}}[
        (center[i] - radius, center[i] + radius) for i in 1:n]
    box = [_iv(lo, hi, bits) for (lo, hi) in box_bounds]
    Hmid = _hessian(model, box, bits)
    midH = BigFloat[_interval_mid(Hmid[i, j], bits) for i in 1:n, j in 1:n]
    Cbig = _big_inverse(midH)
    C = Rational{BigInt}[Rational{BigInt}(Cbig[i, j]) for i in 1:n, j in 1:n]
    _evaluate_root_claims(model, original, translation, normalized, center,
        box_bounds, C, bits, options, provenance, attempts)
end

function _evaluate_root_claims(model::CertificationModel,
        original::Vector{Rational{BigInt}}, translation::Vector{BigInt},
        normalized::Vector{Rational{BigInt}}, center::Vector{Rational{BigInt}},
        box_bounds::Vector{Tuple{Rational{BigInt},Rational{BigInt}}},
        C::Matrix{Rational{BigInt}}, bits::Int, options::ProofOptions,
        provenance::Dict{String,String},
        attempts::Vector{String})
    n = length(center)
    box = [_iv(lo, hi, bits) for (lo, hi) in box_bounds]
    x0 = _point_intervals(center, bits)
    f0 = _gradient(model, x0, bits)
    H = _hessian(model, box, bits)
    Ci = _interval_matrix(C, bits)
    CH = _matrix_product(Ci, H, bits)
    R = Matrix{IA.Interval{BigFloat}}(undef, n, n)
    for j in 1:n, i in 1:n
        R[i, j] = _isub(_iv(i == j ? 1 : 0, bits), CH[i, j], bits)
    end
    CF = _matrix_vector(Ci, f0, bits)
    delta = [_isub(box[i], x0[i], bits) for i in 1:n]
    Rdelta = _matrix_vector(R, delta, bits)
    W = [_iadd(_isub(x0[i], CF[i], bits), Rdelta[i], bits) for i in 1:n]
    q = BigFloat(0; precision=bits)
    for i in 1:n
        row = _iv(0, bits)
        for j in 1:n
            row = _iadd(row, _iabs(R[i, j], bits), bits)
        end
        q = max(q, _hi(row))
    end
    included = all(_lo(W[i]) > _lo(box[i]) && _hi(W[i]) < _hi(box[i]) for i in 1:n)
    contracted = q < 1
    ldl = _interval_ldl(H, bits)
    class = _classify(ldl.signs, n)
    method = ldl.ok ? :direct_ldl : :none
    pivots = ldl.ok ? ldl.pivots : IA.Interval{BigFloat}[]
    status = included && contracted ? :CERTIFIED_ROOT : :INCONCLUSIVE
    reason = status === :CERTIFIED_ROOT ? "strict inclusion and q<1" :
        included ? "contraction bound q is not below one" :
        contracted ? "Krawczyk image is not strictly inside the original box" :
        "root proof was inconclusive"
    return RootResult(status, _model_signature(model), reason, copy(original),
        copy(translation), copy(normalized), copy(center), box, copy(C), f0, H, W,
        Rational{BigInt}(q), bits, options, class, method,
        pivots, copy(provenance), copy(attempts))
end

function _model_signature(model::CertificationModel)
    parts = String[string(size(model.Q, 1), ",", size(model.Q, 2))]
    for A in (model.Q, model.L, model.K)
        for j in axes(A, 2), i in axes(A, 1)
            push!(parts, string(A[i, j]))
        end
    end
    append!(parts, string.(model.phases)); push!(parts, string(model.s))
    bytes2hex(sha256(codeunits(join(parts, "|"))))
end

_rational_bound(x::BigFloat) = Rational{BigInt}(x)

function _contained_claim(recomputed::IA.Interval{BigFloat}, stored::IA.Interval{BigFloat})
    IA.isguaranteed(recomputed) && IA.decoration(recomputed) == IA.com &&
    IA.isguaranteed(stored) && IA.decoration(stored) == IA.com &&
    _rational_bound(_lo(stored)) <= _rational_bound(_lo(recomputed)) &&
    _rational_bound(_hi(recomputed)) <= _rational_bound(_hi(stored))
end

function _revalidate_root(model::CertificationModel, root::RootResult, bits::Int)
    n = size(model.Q, 1)
    root.status === :CERTIFIED_ROOT || return false
    root.model_signature == _model_signature(model) || return false
    length(root.original_seed) == n && length(root.normalized_seed) == n &&
        length(root.translation) == n && length(root.center) == n &&
        length(root.box) == n && size(root.preconditioner) == (n,n) || return false
    root.normalized_seed == [root.original_seed[i] - root.translation[i] for i in 1:n] || return false
    all(-1//2 <= root.normalized_seed[i] < 1//2 for i in 1:n) || return false
    all(floor(BigInt, root.original_seed[i] + 1//2) == root.translation[i] for i in 1:n) || return false
    bounds = Tuple{Rational{BigInt},Rational{BigInt}}[]
    for i in 1:n
        _isvalidproof(root.box[i]) || return false
        lo, hi = _rational_bound(_lo(root.box[i])), _rational_bound(_hi(root.box[i]))
        0 < hi-lo < 1 || return false
        lo < root.center[i] < hi || return false
        push!(bounds, (lo, hi))
    end
    root.q_upper < 1 || return false
    length(root.F_center) == n && size(root.H_box) == (n,n) && length(root.W) == n || return false
    evaluated = try
        _evaluate_root_claims(model, root.original_seed, root.translation,
            root.normalized_seed, root.center, bounds, root.preconditioner,
            bits, root.options, root.candidate_provenance, root.attempts)
    catch e
        e isa SingularException || e isa DomainError || e isa OverflowError || rethrow()
        return false
    end
    evaluated.status === :CERTIFIED_ROOT || return false
    evaluated.q_upper <= root.q_upper || return false
    all(_contained_claim(evaluated.F_center[i], root.F_center[i]) for i in 1:n) || return false
    all(_contained_claim(evaluated.H_box[i,j], root.H_box[i,j]) for j in 1:n for i in 1:n) || return false
    all(_contained_claim(evaluated.W[i], root.W[i]) for i in 1:n) || return false
    all(_lo(evaluated.W[i]) > _lo(root.box[i]) && _hi(evaluated.W[i]) < _hi(root.box[i]) for i in 1:n) || return false
    if root.hessian_method === :direct_ldl
        length(root.hessian_pivots) == n && evaluated.hessian_method === :direct_ldl || return false
        all(_contained_claim(evaluated.hessian_pivots[i], root.hessian_pivots[i]) for i in 1:n) || return false
        evaluated.hessian_classification === root.hessian_classification || return false
    elseif root.hessian_method === :none
        isempty(root.hessian_pivots) && root.hessian_classification === :UNRESOLVED || return false
    else
        return false
    end
    true
end

function _normalize_seed(seed::Vector{Rational{BigInt}})
    translation = BigInt[floor(BigInt, x + 1//2) for x in seed]
    normalized = Rational{BigInt}[seed[i] - translation[i] for i in eachindex(seed)]
    return translation, normalized
end

function _provenance(value)
    value === nothing && return Dict("status"=>"unknown")
    value isa NamedTuple && (value = Dict(string(k)=>v for (k,v) in pairs(value)))
    value isa AbstractDict || throw(ArgumentError("candidate_provenance must be a record"))
    method = get(value, "method", get(value, :method, nothing))
    source = get(value, "source_ref", get(value, :source_ref, nothing))
    method isa AbstractString && !isempty(method) || throw(ArgumentError("provenance method is required"))
    source isa AbstractString && !isempty(source) || throw(ArgumentError("provenance source_ref is required"))
    Dict("status"=>"supplied", "method"=>String(method), "source_ref"=>String(source))
end

function certify_stationary_point(model::CertificationModel, theta0;
        options=ProofOptions(), candidate_provenance=nothing)
    options isa ProofOptions || throw(ArgumentError("options must be ProofOptions"))
    seed = _exact_seed(theta0, size(model.Q, 1))
    translation, normalized = _normalize_seed(seed)
    provenance = _provenance(candidate_provenance)
    (length(seed) <= options.max_dimension && size(model.Q, 2) <= options.max_terms) ||
        return _empty_root(:UNSUPPORTED, "RESOURCE_LIMIT", seed)
    _backend_supported() || return _empty_root(:UNSUPPORTED, "INTERVAL_BACKEND_VERSION", seed)
    attempts = String[]
    retained = nothing
    for bits in options.precisions
        witness_for_precision = _with_precision(bits) do
            x = _proposal(model, normalized, options, bits)
            center = Rational{BigInt}[Rational{BigInt}(v) for v in x]
            best = nothing
            for radius in options.box_radii
                push!(attempts, "precision=$bits radius=$radius")
                witness = try
                    _root_witness(model, seed, translation, normalized, center,
                                  radius, bits, options, provenance, attempts)
                catch e
                    (e isa SingularException || e isa DivideError ||
                     e isa ProofArithmeticError || e isa DomainError) || rethrow()
                    nothing
                end
                witness === nothing && continue
                best === nothing && (best = witness)
                witness.status === :CERTIFIED_ROOT && return witness
            end
            best
        end
        retained === nothing && witness_for_precision !== nothing && (retained = witness_for_precision)
        witness_for_precision !== nothing && witness_for_precision.status === :CERTIFIED_ROOT &&
            return witness_for_precision
    end
    retained === nothing ? _empty_root(:INCONCLUSIVE, "no finite root witness", seed) :
        RootResult(:INCONCLUSIVE, retained.model_signature, retained.reason, retained.original_seed,
            retained.translation, retained.normalized_seed, retained.center,
            retained.box, retained.preconditioner, retained.F_center, retained.H_box,
            retained.W, retained.q_upper, retained.precision, retained.options,
            retained.hessian_classification, retained.hessian_method,
            retained.hessian_pivots, provenance, attempts)
end
