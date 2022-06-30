module generate
###############################
##### Pseudo-Geometric data ###
###############################

function pseudo_Q(h11,tri,cy=1)
    Q = vcat(Matrix{Float64}(I(h11)),rand(-5.:5.,4,h11))
    return vcat(Q,hcat([Q[i,:]-Q[j,:] for i=1:size(Q,1)-1 for j=i+1:size(Q,1)]...)')
end

function pseudo_K(h11,tri,cy=1)
    K::Matrix{Float64} = rand(h11,h11)
    K = 4* 0.5 * (K+transpose(K)) + 2 .* I(h11)
    return Hermitian(K)
end

function pseudo_L(h11,tri,cy=1)
    Llogprime::Vector{Float64} = sort(vcat([0,[-(4*(j-1)) for j=2:h11+4]...]),rev=true)
    Lsignh11::Vector{Float64} = rand(Uniform(0,100),h11-1)
    Lsign4::Vector{Float64} = rand(Uniform(-100,100),4)
    Lsignprime::Vector{Float64} = vcat(1.,Lsignh11..., Lsign4...)
    Llogcross::Vector{Float64} = [Llogprime[i] + Llogprime[j] for i=1:size(Llogprime,1)-1 for j=i+1:size(Llogprime,1)]
    Lsigncross::Vector{Float64} = [Lsignprime[i] * Lsignprime[j] for i=1:size(Lsignprime,1)-1 for j=i+1:size(Lsignprime,1)]
    Llogtemp::Vector{Float64} = vcat(Llogprime...,Llogcross...)
    Lsigntemp::Vector{Float64} = vcat(Lsignprime...,Lsigncross...)
    Ltemp::Vector{ArbFloat} = ArbFloat.(Lsigntemp) .* ArbFloat(10.) .^ ArbFloat.(Llogtemp)
    return Ltemp
end

function pseudo_Llog(h11,tri,cy=1)
    L1 = [1 1]
    L2 = vcat([[1 -(4*(j-1))] for j=2:h11+4]...)
    L3 = vcat([[sign(rand(Uniform(-100. *h11,100. *h11))) -(4*(j-1))+log10(abs(rand(Uniform(-100. *h11,100. *h11))))]
     for j=h11+5:h11+4+binomial(h11+4,2)]...)
    L4 = @.(log10(abs(L3)))
    return vcat(L1,L2,L3)
end



end