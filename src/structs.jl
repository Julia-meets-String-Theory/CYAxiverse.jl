module structs

struct GeometryIndex{T<:Integer}
    h11::T
    tri::T
    cy::T
end

struct LQLinearlyIndependent
    Qtilde::Matrix{Int}
    Qbar::Matrix{Int}
    Lbar::Matrix{Float64}
    Ltilde::Matrix{Float64}
end

struct Projector{T<:Matrix{Real}}
    Π::T
    Πperp::T
end

struct CanonicalQBasis
    Qhat::Matrix{Int}
    Qbar::Matrix{Int}
    Lhat ::Matrix{Float64}
    Lbar ::Matrix{Float64}
    α::Matrix{Rational}
end


end