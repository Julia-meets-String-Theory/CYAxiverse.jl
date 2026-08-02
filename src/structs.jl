module structs
using AbstractTrees
using SparseArrays
using LinearAlgebra
using AbstractTrees: isroot, parent

"""
    GeometryIndex{T<:Integer}

Identifies geometry by `h11, polytope, frst`
"""
Base.@kwdef struct GeometryIndex{T<:Integer}
    h11::T
    polytope::T
    frst::T=1
end

struct TopologicalData
    points::Matrix{Int}
    simplices::Matrix{Int}
end

struct GeometricData
    tip_prefactor::Vector{Float64}
    τ_volumes::Vector{Float64}
    h21::Integer
    cy_volume::Float64
    glsm_charges::Matrix{Int}
    basis::Vector{Int}
    tip::Vector{Float64}
    kinv::Matrix{Float64}
    hilbert_basis::Matrix{Int}
end

struct AxionPotential
    L::Matrix{Float64}
    Q::Matrix{Int}
    K::Hermitian{Float64, Matrix{Float64}}
end

"""
    QuarticComponentDiagnostics

Float64 cancellation diagnostics for one family of signed quartic components.
`orders_lost` estimates cancellation in orders of magnitude,
`digits_remaining` estimates the remaining decimal digits, `reliable` marks
components with at least three estimated decimal digits, and `exact_zero`
marks a zero result from the Float64 contraction.
"""
struct QuarticComponentDiagnostics
    orders_lost::Vector{Float64}
    digits_remaining::Vector{Float64}
    reliable::BitVector
    exact_zero::BitVector
end

"""
    QuarticDiagnostics

Cancellation diagnostics aligned respectively with `λself`, `λ31`, and
`λ22`. Produced only when `pq_spectrum(...; quartic_diagnostics=true)` is
requested.
"""
struct QuarticDiagnostics
    self::QuarticComponentDiagnostics
    three_one::QuarticComponentDiagnostics
    two_two::QuarticComponentDiagnostics
end

"""
    MassBasisDiagnostics

Float64 diagnostics for a PQ leading-Hessian mass basis. `eigenpair_residuals`
are relative residuals, `nearest_relative_gaps` quantify the nearest spectral
neighbour of each mode, and `orthogonality_error` is the infinity-norm error
of the basis Gram matrix from the identity.
"""
struct MassBasisDiagnostics
    eigenpair_residuals::Vector{Float64}
    nearest_relative_gaps::Vector{Float64}
    orthogonality_error::Float64
end

"""
    InstantonHierarchyDiagnostics

Cheap physics-informed diagnostics of the instanton scale hierarchy.
`leading_log_gap` is the gap between the two largest entries of `L[2, :]`,
and `log_scale_span` is its full range. `heuristic_strong_hierarchy` uses the
provisional thresholds documented by `pq_spectrum`; it is a screening flag,
not a numerical accuracy certificate.
"""
struct InstantonHierarchyDiagnostics
    leading_log_gap::Float64
    log_scale_span::Float64
    heuristic_strong_hierarchy::Bool
end

"""
    PhysicalAxionSpectrum

High-precision PQ leading-Hessian result restricted to modes above a physical
mass threshold. `mode_indices` are zero-based indices in the complete sorted
leading-Hessian spectrum; `eigenvectors` are the corresponding canonical-field
eigenvectors. Quartics include every instanton but only retained physical-mode
indices.
"""
struct PhysicalAxionSpectrum
    m::Vector{Float64}
    mode_indices::Vector{Int}
    eigenvectors::Matrix{Float64}
    λselfsign::Vector{Int}
    λself::Vector{Float64}
    λ31_i::Matrix{Int}
    λ31sign::Vector{Int}
    λ31::Vector{Float64}
    λ22_i::Matrix{Int}
    λ22sign::Vector{Int}
    λ22::Vector{Float64}
    threshold_log10::Float64
    prec::Int
end

"""
    AxionSpectrum

PQ-spectrum result. `m` is expressed in the basis selected by
`mixing_correction`; `f` and `fK` retain their PQ/kinetic-basis definitions.
The signed base-10 logarithms `λself`, `λ31`, and `λ22` are accompanied by
their signs and zero-based index matrices. `quartic_diagnostics` is `nothing`
by default and is a [`QuarticDiagnostics`](@ref) only when explicitly
requested.
`mass_basis_diagnostics` is `nothing` by default and is a
[`MassBasisDiagnostics`](@ref) when requested for a leading-Hessian mass
basis.
`instanton_hierarchy` is `nothing` by default and is an
[`InstantonHierarchyDiagnostics`](@ref) when explicitly requested. It is not
a substitute for numerical mass-basis diagnostics.
"""
struct AxionSpectrum
    m::Vector{Float64}
    f::Vector{Float64}
    fK::Vector{Float64}
    λselfsign::Vector{Int}
    λself::Vector{Float64}
    λ31_i::Matrix{Int}
    λ31sign::Vector{Int}
    λ31::Vector{Float64}
    λ22_i::Matrix{Int}
    λ22sign::Vector{Int}
    λ22::Vector{Float64}
    quartic_diagnostics::Union{Nothing, QuarticDiagnostics}
    mass_basis_diagnostics::Union{Nothing, MassBasisDiagnostics}
    instanton_hierarchy::Union{Nothing, InstantonHierarchyDiagnostics}
end

Base.@kwdef struct IndexedAxionSpectrum{T<:Float64}
    h11::Int
    polytope::Int
    frst::Int
    m::Vector{T}
    f::Vector{T}
    fK::Vector{T}
    function IndexedAxionSpectrum(geom_idx::GeometryIndex, spectrum::AxionSpectrum)
        IndexedAxionSpectrum(geom_idx.h11, geom_idx.polytope, geom_idx.frst, spectrum.m, spectrum.f, spectrum.fK)
    end
end


struct LQLinearlyIndependent
    Qtilde::Matrix{Int}
    Qbar::Matrix{Int}
    Lbar::Matrix{Float64}
    Ltilde::Matrix{Float64}
end

struct Projector{T<:Union{Rational, Float64, Integer}}
    Π::Matrix{T}
    Πperp::Matrix{T}
end

struct ProjectedQ
    Ωperp::SparseMatrixCSC
    Ωparallel::SparseMatrixCSC
end


struct CanonicalQBasis
    Qhat::Matrix{Int}
    Qbar::Matrix{Int}
    Lhat ::Matrix{Float64}
    Lbar ::Matrix{Float64}
end

struct Canonicalα
    Qhat::Matrix{Int}
    Qbar::Matrix{Int}
    Lhat ::Matrix{Float64}
    Lbar ::Matrix{Float64}
    α::Matrix{Rational}
    α_complete::Matrix{Rational}
    αrowmask::Vector{Bool}
    αcolmask::Vector{Bool}
end

struct ReducedPotential
    αeff::Matrix{Rational}
    Lreduced::Matrix{Float64}
end

struct Solver1D
    search_domain
    Q::Vector
    Llog::Vector
    Lsign::Vector
    det_QTilde
    phases::Vector
    Z
    inv_symmetries
    det_Sym
end
struct RationalQSNF
    Tparallel::Matrix{Rational}
    θparallel::Matrix{Rational}
end

struct BasisSNF
    volume::Number
    basis::Matrix
    id_coords::Matrix
end

struct SolverND
    samples
    Q::Matrix
    Llog::Vector
    Lsign::Vector
    det_QTilde
    phases::Vector
    Z
    inv_symmetries
    det_Sym
end

struct Min_JLM_ND
    N_min::Integer
    min_coords::Matrix{Float64}
    extra_rows::Integer
    det_QTilde::Integer
end

struct Min_JLM_Square
    N_min::Integer
    det_QTilde::Integer
end


struct Min_JLM_1D
    N_min::Integer
    min_coords::Vector{Float64}
    extra_rows::Integer
    det_QTilde::Integer
end
#######################
# ParentTrackIterator #
#######################
### from: https://discourse.julialang.org/t/help-design-a-node-for-a-tree/67444/10 ###



######
# Tree
######

struct MyTree{D}
    data::D
    parent_min::Union{Nothing,MyTree{D}}
    parent_phase::Union{Nothing,MyTree{D}}
    subtrees::Vector{MyTree{D}}

    function MyTree{D}(d::D, ::Nothing, v::AbstractVector{MyTree{D}}) where D
        new{D}(d, nothing, v)
    end
    function MyTree{D}(d::D, parent_min::MyTree{D}, v::AbstractVector{MyTree{D}}) where D
        ret = new{D}(d, parent_min, v)
        push!(parent_min.subtrees, ret)
        ret
    end
end
MyTree(d::T, parent=nothing, v=MyTree{T}[]) where T = MyTree{T}(d, parent, v)
Base.eltype(::Type{MyTree{T}}) where T = T 

AbstractTrees.children(t::MyTree) = t.subtrees
AbstractTrees.parent(t::MyTree) = t.parent_min
AbstractTrees.isroot(t::MyTree) = parent(t) === nothing

Base.show(io::IO, t::MyTree) = print(io, "MyTree{D}(", t.data, ')')

struct ParentTrack{T}
    tree::T
end

Base.IteratorEltype(::Type{<:ParentTrack}) = Base.HasEltype()
Base.IteratorSize(::Type{<:ParentTrack}) = Base.SizeUnknown()
Base.eltype(::Type{ParentTrack{MyTree{T}}}) where T = Vector{eltype(T)}

Base.iterate(pt::ParentTrack{MyTree{T}}) where T = iterate(pt, (MyTree{T}[], [pt.tree]))
function Base.iterate(_::ParentTrack, (parents, toProcess))
    isempty(toProcess) && return nothing
    local el
    # push work items until we can't anymore
    while true
        el = pop!(toProcess)
        children = el.subtrees
        push!(parents, el)
        if isempty(children)
            break
        else
            append!(toProcess, children)
        end
    end
    # we're in a leaf

    # get our return value and remove ourselves
    c = map(x -> x.data, parents)
    pop!(parents)
    if !isempty(toProcess) && last(toProcess).parent_min != el.parent_min
        pop!(parents) # pop the parent
    end
    return c, (parents, toProcess)
end

end
