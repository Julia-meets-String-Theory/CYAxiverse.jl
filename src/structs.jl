module structs

Base.@kwdef struct GeometryIndex{T<:Integer}
    h11::T
    tri::T
    cy::T=1
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

struct ProjectedQ{T<:Real}
    Ωperp::Matrix{T}
    Ωparallel::Matrix{T}
end


struct CanonicalQBasis
    Qhat::Matrix{Int}
    Qbar::Matrix{Int}
    Lhat ::Matrix{Float64}
    Lbar ::Matrix{Float64}
    α::Matrix{Rational}
end


end