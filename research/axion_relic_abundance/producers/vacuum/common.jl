# Shared helpers for the vacuum-repair stage (no writes to any dataset).
using CYAxiverse, LinearAlgebra
const Gc = CYAxiverse.generate

const _c = Gc.constants()
const THR = Float64(log10(_c["Hubble"]))                                       # production H0 threshold, log10 eV
const OFFSET = 9 + Float64(log10(_c["MPlanck"])) + Float64(_c["log2π"])        # log10 m[eV] = 0.5 log10(lambda_W) + OFFSET

tojson(x::AbstractString) = "\"" * replace(x, "\\" => "\\\\", "\"" => "\\\"") * "\""
tojson(x::Symbol) = tojson(string(x))
tojson(x::Bool) = x ? "true" : "false"
tojson(x::Integer) = string(x)
tojson(x::AbstractFloat) = isfinite(x) ? repr(Float64(x)) : (isnan(x) ? "null" : (x > 0 ? "1e999" : "-1e999"))
tojson(x::AbstractVector) = "[" * join(tojson.(x), ",") * "]"
tojson(x::Tuple) = tojson(collect(x))
tojson(x::AbstractDict) = "{" * join([tojson(string(k)) * ":" * tojson(v) for (k, v) in x], ",") * "}"
tojson(x::Nothing) = "null"

geom_id(h, pos) = "h11_" * lpad(h, 3, '0') * "/np_" * lpad(pos, 7, '0') * "/cy_0000001"
spectra_file(root, h, pos) = joinpath(root, geom_id(h, pos), "cyax.h5")

"""log10(m/eV) of a leading-Hessian (W) eigenvalue; sign ignored."""
mass_log10(v) = Float64(0.5 * log10(abs(v))) + OFFSET

"""Generated potentials hold n single terms followed by n(n-1)/2 pair differences; return n (or -1)."""
function single_term_count(N)
    r = isqrt(8N + 1); r^2 == 8N + 1 || return -1
    (r - 1) ÷ 2
end

"""Q/L column indices of the LQtilde-selected leading instantons, in LQtilde order (identical selection code)."""
function leading_columns(Q, L, Qt, Lt)
    h11 = size(Q, 1)
    perm = sortperm(@view(L[2, :]), rev=true)
    mask = fill(false, size(Q, 2))
    Gc.leading_independent_mask!(mask, Q, perm, zeros(Float64, h11, h11), zeros(Float64, h11))
    cols = [i for i in perm if mask[i]]
    @assert Q[:, cols] == Qt && L[:, cols] == Lt
    cols
end

"""
Exact leading-branch minima: theta* = Qtilde^{-T} (mask/2 + n) with mask_i = 1 iff the leading amplitude sign is
negative and n running over the |det Qtilde| lattice cosets (repository `leading_lattice_offsets`). For every copy
return the exact rational phase q_a . theta* mod 1 of every instanton a, whether all phases lie in {0, 1/2}
(then theta* is an exact critical point of the FULL cosine potential), and the effective sign cos(2 pi phase).
"""
function exact_leading_minima(Q, L; max_copies::Int=4096)
    lq = Gc.LQtilde(Q, L); Qt, Lt = lq.Qtilde, lq.Ltilde
    h11 = size(Qt, 1)
    d = abs(Gc._exact_integer_determinant(Qt))
    Qr = Rational{BigInt}.(Qt)
    alpha = Qr \ Rational{BigInt}.(Q)                       # alpha[:, a] = Qtilde^{-1} q_a (exact)
    mask = [Lt[1, i] < 0 ? 1 // 2 : 0 // 1 for i in 1:h11]
    offs = d <= max_copies ? Gc.leading_lattice_offsets(lq) : zeros(h11, 1)
    copies = NamedTuple[]
    QtT_inv = inv(transpose(Qr))
    for j in axes(offs, 2)
        n = round.(BigInt, transpose(Float64.(Qt)) * offs[:, j])  # integer coset representative
        base = Rational{BigInt}.(mask) .+ n
        theta = mod.(QtT_inv * base, 1)
        phase = mod.(vec(transpose(alpha) * base), 1)          # q_a . theta* mod 1, exact
        onhalf = [p == 0 || p == 1 // 2 for p in phase]
        eff = [p == 0 ? 1 : (p == 1 // 2 ? -1 : 0) for p in phase]
        offgrid = findall(.!onhalf)
        push!(copies, (; theta, phase, all_half=all(onhalf), n_offgrid=length(offgrid),
            offgrid_max_log10=isempty(offgrid) ? -Inf : maximum(L[2, offgrid]),
            n_flipped=count(==(-1), eff), eff))
    end
    (; det=Int(d), copies, lq)
end
