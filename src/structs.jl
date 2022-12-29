module structs
using AbstractTrees
using SparseArrays
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

struct AxionPotential
    L::Matrix{Float64}
    Q::Matrix{Int}
    K::Matrix{Float64}
end

struct LQLinearlyIndependent
    Qtilde::Matrix{Int}
    Qbar::Matrix{Int}
    Lbar::Matrix{Float64}
    Ltilde::Matrix{Float64}
end

struct Projector
    Π::Matrix{Rational}
    Πperp::Matrix{Rational}
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
    α::Matrix{Rational}
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