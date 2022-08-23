module generate

using HDF5
using LinearAlgebra
using ArbNumerics, Tullio, LoopVectorization, Nemo
using GenericLinearAlgebra
using Distributions
using TimerOutputs

using ..filestructure: cyax_file, minfile, present_dir
using ..read: potential

#################
### Constant ####
#################

"""
    constants()

Loads constants:\n
- Reduced Planck Mass = 2.435 × 10^18
- Hubble = 2.13 × 0.7 × 10^-33
- log2pi = log10(2π)
as `Dict{String,ArbFloat}`\n
#Examples
```julia-repl
julia> const_data = CYAxiverse.generate.constants()
Dict{String, ArbNumerics.ArbFloat{128}} with 3 entries:
  "MPlanck" => 2435000000000000000.0
  "log2π"   => 0.7981798683581150521959557408991
  "Hubble"  => 1.490999999999999999287243983194e-33
```
"""
function constants()
    mplanck_r::ArbFloat = ArbFloat(2.435e18)
    hubble::ArbFloat = ArbFloat(2.13*0.7*1e-33)
    log2pi::ArbFloat = ArbFloat(log10(2π))
    return Dict("MPlanck" => mplanck_r, "Hubble" => hubble, "log2π" => log2pi)
end


###############################
##### Pseudo-Geometric data ###
###############################

"""
    pseudo_Q(h11,tri,cy=1)

Randomly generates an instanton charge matrix that takes the same form as those found in the KS Axiverse, namely `I(h11)` with 4 randomly filled rows and the cross-terms, i.e. an h11+4+C(h11+4,2) × h11 integer matrix.
#Examples
```julia-repl
julia> CYAxiverse.generate.pseudo_Q(4,10,1)
36×4 Matrix{Int64}:
  1   0   0   0
  0   1   0   0
  0   0   1   0
  0   0   0   1
  1   4  -3   5
 -5  -4  -2   4
  4   5   3  -2
 -5   2  -3  -3
  ⋮
 -9  -9  -5   6
  0  -6   1   7
  9   3   6   1
```
"""
function pseudo_Q(h11::Int,tri::Int,cy::Int=1)
    Q = vcat(Matrix{Int}(I(h11)),rand(-5:5,4,h11))
    return vcat(Q,hcat([Q[i,:]-Q[j,:] for i=1:size(Q,1)-1 for j=i+1:size(Q,1)]...)')
end

"""
    pseudo_K(h11,tri,cy=1)

Randomly generates an h11 × h11 Hermitian matrix with positive gauss_difffinite eigenvalues
#Examples
```julia-repl
julia> K = CYAxiverse.generate.pseudo_K(4,10,1)
4×4 Hermitian{Float64, Matrix{Float64}}:
 2.64578  2.61012  0.91203  2.27339
 2.61012  3.89684  2.22451  1.93356
 0.91203  2.22451  2.94717  1.58126
 2.27339  1.93356  1.58126  4.85208

julia> eigen(K).values
4-element Vector{Float64}:
 0.17629073145135896
 1.8632009739875723
 2.7425362219513487
 9.559840749713599
```
"""
function pseudo_K(h11::Int,tri::Int,cy::Int=1)
    K::Matrix{Float64} = rand(h11,h11)
    K = 4* 0.5 * (K+transpose(K)) + 2 .* I(h11)
    while minimum(eigen(K).values) < 0.
        K = rand(h11,h11)
        K = 4* 0.5 * (K+transpose(K)) + 2 .* I(h11)
    end
    return Hermitian(K)
end

"""
    pseudo_L(h11,tri,cy=1;log=true)

Randomly generates a h11+4+C(h11+4,2)-length hierarchical list of instanton scales, similar to those found in the KS Axiverse.  Option for (sign,log10) or full precision.
#Examples
```julia-repl
julia> CYAxiverse.generate.pseudo_L(4,10)
36×2 Matrix{Float64}:
  1.0     0.0
  1.0    -4.0
  1.0    -8.0
  1.0   -12.0
  1.0   -16.0
  1.0   -20.0
  1.0   -24.0
  1.0   -28.0
 -1.0   -29.4916
  1.0   -33.8515
  ⋮
  1.0  -133.665
 -1.0  -138.951

julia> CYAxiverse.generate.pseudo_L(4,10,log=false)
36-element Vector{ArbNumerics.ArbFloat}:
  1.0
  0.0001
  1.0e-8
  1.0e-12
  1.0e-16
  1.0e-20
  1.0e-24
  1.0e-28
 -1.462574279422558833057690597964e-31
 -2.381595397961591074099629406235e-34
  ⋮
  3.796809523142314798130344022481e-134
 -3.173000613781491329619833894919e-138
```
"""
function pseudo_L(h11::Int,tri::Int,cy::Int=1;log::Bool=true)
    L1::Matrix{Float64} = [1. 0.]
    L2::Matrix{Float64} = vcat([[1. -(4. *(j-1.))] for j=2.:h11+4.]...)
    L3::Matrix{Float64} = vcat([[sign(rand(Uniform(-100. *h11,100. *h11))) -(4*(j-1))+log10(abs(rand(Uniform(-100. *h11,100. *h11))))]
     for j=h11+5:h11+4+binomial(h11+4,2)]...)
    L4::Matrix{Float64} = @.(log10(abs(L3)))
    L::Matrix{Float64} = vcat(L1,L2,L3)
    L = hcat(sign.(L[:,1]), log10.(abs.(L[:,1])) .+ L[:,2])
    if log == 1
        return L
    else
        Ltemp::Vector{ArbFloat} = ArbFloat.(L[:,1]) .* ArbFloat(10.) .^ ArbFloat.(L[:,2])
        return Ltemp
    end
end

##############################
#### Computing Spectra #######
##############################

"""
    gauss_sum(z)

Computes the addition of 2 numbers in (natural) log-space using the gauss_difffinition [here](https://en.wikipedia.org/wiki/Gaussian_logarithm).
#Examples
```julia-repl
julia> CYAxiverse.generate.gauss_sum(10.)
10.000045398899218

julia> CYAxiverse.generate.gauss_sum(1000.)
1000.0
```
"""
function gauss_sum(z::Float64)
    log2 = log(2)
    if abs(z)>600.
        return 0.5*z +abs(0.5*z)
    else
        return log2 + 0.5*z + log(cosh(0.5*z))
    end
end
"""
    gauss_diff(z)

Computes the difference of 2 numbers in (natural) log-space using the gauss_difffinition [here](https://en.wikipedia.org/wiki/Gaussian_logarithm).
#Examples
```julia-repl
julia> CYAxiverse.generate.gauss_diff(10.)
9.99995459903963

julia> CYAxiverse.generate.gauss_diff(1000.)
1000.0
```
"""
function gauss_diff(z::Float64)
    log2 = log(2)
    if abs(z)>600.
        return 0.5*z +abs(0.5*z)
    else
        return log2 + 0.5*z + log(abs(sinh(0.5*z)))
    end
end

"""
    gauss_log_split(sign, log)

Algorithm to compute Gaussian logarithms, as gauss_difftailed [here](https://en.wikipedia.org/wiki/Gaussian_logarithm).
#Examples
```julia-repl
julia> CYAxiverse.generate.gauss_diff(10.)
9.99995459903963

julia> CYAxiverse.generate.gauss_diff(1000.)
1000.0
```
"""
function gauss_log_split(sb::Vector{Int},logb::Vector{Float64})
    # loga = log(|A|); logb = log(|B|); sa = sign(A); sb = sign(B)
    temp = hcat(sb,logb)
    temp = temp[sortperm(temp[:,2]),:]
    sb::Vector{Int} = temp[:,1]
    logb::Vector{Float64} = temp[:,2]
    i = 1
    sa = sb[i]
    loga = logb[i]
    while i < size(sb,1)
#         println(i)
#         println([sa,sb[i+1],loga, logb[i+1]])
        if (sa==0 && sb[i+1]==0) ## A == B == 0
        elseif sa==0 ## A==0 --> B
            sa = sb[i+1]
            loga = logb[i+1]
        elseif sb[i+1]==0 ## B==0 --> A
        elseif (sa<0 && sb[i+1]>0) ## B-A
            if loga<logb[i+1] ## |A|<|B|
                sa = 1
                loga = logb[i+1]+gauss_diff(loga-logb[i+1])
            elseif loga == logb[i+1]
                sa = 0
                loga = 0
            else ## |A|>|B|
                sa = -1
                loga = logb[i+1]+gauss_diff(loga-logb[i+1])
            end
        elseif (sa>0 && sb[i+1]<0) ## A-B
            if loga>logb[i+1] ## |A|>|B|
                sa =1
                loga = loga+gauss_diff(-loga+logb[i+1])
            elseif loga == logb[i+1]
                sa = 0
                loga = 0
            else ## |A|<|B|
                sa = -1
                loga = loga+gauss_diff(-loga+logb[i+1])
            end
        elseif (sa<0 && sb[i+1]<0) ## -A-B
            sa = -1
            loga = loga + gauss_sum(-loga+logb[i+1])
        else ## A+B
            sa = 1
            loga = loga + gauss_sum(-loga+logb[i+1])
        end
        i+=1
    end
    return Int(sa), Float64(loga)
end

function gauss_log(sb,logb)
    if size(sb[sb .== 0.],1) == size(sb,1)
        return 0,-Inf
    elseif size(sb[sb .> 0.],1) == 0
        test = -1
#     elseif size(sb[sb .< 0.],1) == 0
#         test = 1
    else
        test = 1
    end
    temp = hcat(sb,logb)
    signed_mask::Vector{Bool} = temp[:,1] .== test
    temp1 = temp[signed_mask,:]
    temp1 = temp1[sortperm(temp1[:,2]),:]
    sb::Vector{Int} = temp1[:,1]
    logb::Vector{Float64} = temp1[:,2]
    sa1::Int,loga1::Float64 = gauss_log_split(sb,logb)
    if size(temp1,1) != size(temp,1)
        signed_mask = Bool.(true .- signed_mask)
        temp2 = temp[signed_mask,:]
        temp2 = temp2[sortperm(temp2[:,2]),:]
        sba::Vector{Int} = temp2[:,1]
        logba::Vector{Float64} = temp2[:,2]
        sa2::Int,loga2::Float64 = gauss_log_split(sba,logba)
        sa3::Vector{Int} = vcat(sa1,sa2)
        loga3::Vector{Float64} = vcat(loga1,loga2)
        sa::Int,loga = gauss_log_split(sa3,loga3)
        return Int(sa),Float64(loga)
    else
        return Int(sa1), Float64(loga1)
    end
end

"""
    hp_spectrum(K,L,Q; prec=5_000)

Uses potential data generated by CYTools (or randomly generated) to compute axion spectra -- masses, quartic couplings and gauss_diffcay constants -- to high precision.
#Examples
```julia-repl
julia> const_data = CYAxiverse.generate.constants()
Dict{String, ArbNumerics.ArbFloat{128}} with 3 entries:
  "MPlanck" => 2435000000000000000.0
  "log2π"   => 0.7981798683581150521959557408991
  "Hubble"  => 1.490999999999999999287243983194e-33
```
"""
function hp_spectrum(K::Hermitian{Float64, Matrix{Float64}}, L::Matrix{Float64}, Q::Matrix{Int}; prec=5_000)
    @assert size(Q,1) == size(L,1) && size(Q,2) == size(K,1)
    setprecision(ArbFloat,digits=prec)
    h11::Int = size(K,1)
    Lh::Vector{ArbFloat}, Qtest::Matrix{ArbFloat} = L[:,1] .* ArbFloat(10.) .^L[:,2], ArbFloat.(Q)
    #Compute Hessian (in lattice basis)
    grad2::Matrix{ArbFloat} = zeros(ArbFloat,(h11,h11))
    hind1::Vector{Vector{Int64}} = [[x,y]::Vector{Int64} for x=1:h11,y=1:h11 if x>=y]
    grad2_temp::Vector{ArbFloat} = zeros(ArbFloat,size(hind1,1))
#     Lh::Vector{ArbFloat} = zeros(ArbFloat,size(Ltemp,1))
#     @inbounds for i=1:size(Lh,1)
#         Lh[i] = Ltemp[i,1] .* Ltemp[i,2] .* 10. .^ Ltemp[i,3]
#     end
    
    grad2_temp1::Matrix{ArbFloat} = zeros(ArbFloat,size(Lh,1),size(hind1,1))
#     xh::Vector{ArbFloat} = x(h11,tri,cy)
    @tullio grad2_temp1[c,k] = @inbounds(begin
    i,j = hind1[k]
            Qtest[c,i] * Qtest[c,j] end) grad=false fastmath=false
    @tullio grad2_temp[k] = grad2_temp1[c,k] * Lh[c]
    @inbounds for i=1:size(hind1,1)
        j,k = hind1[i]
        grad2[j,k] = grad2_temp[i]
    end
    hessfull = Hermitian(grad2 + transpose(grad2) - Diagonal(grad2))
    Lh = zeros(3)
    #Compute QM using generalised eigengauss_diffcomposition (but keep fK)
    Ktest = Hermitian(ArbFloat.(K))
    fK::Vector{Float64} = Float64.(log10.(sqrt.(eigen(Ktest).values)))
    Vls::Vector{ArbFloat},Tls::Matrix{ArbFloat} = eigen(hessfull, Ktest)
    Hsign::Vector{Int64} = @.(sign(Vls))
    Hvals::Vector{Float64} = @.(log10(sqrt(abs(Vls))))
    QMs::Matrix{ArbFloat} = similar(Qtest)
    multH(M,N) = @tullio fastmath=false grad=false R[c,i] := M[c,j] * N[j,i]
    QMs = multH(Qtest,Tls)
    signQMs::Matrix{Int64} = @.(Int(sign(QMs)))
    logQMs::Matrix{Float64} = @.(Float64(log(abs(QMs))))
    
    #Clear memory
    Vls = zeros(ArbFloat,1)
    Tls = zeros(ArbFloat,1,1)
    QMs = zeros(ArbFloat,1,1)
    Qtest = zeros(ArbFloat,1,1)
    hessfull = zeros(ArbFloat,1,1)
    grad2_temp1 = zeros(ArbFloat,1,1)
    grad2_temp = zeros(ArbFloat,1)
    grad2 = zeros(ArbFloat,1,1)
    Ktest = zeros(ArbFloat,1,1)
#     GC.gc()
    
    #Generate quartics in logspace
    signL::Vector{Int}, logL::Vector{Float64} = L[:,1], L[:,2]
    #Compute quartics
    qindq31::Vector{Vector{Int64}} = [[x,x,x,y]::Vector{Int64} for x=1:h11,y=1:h11 if x!=y]
    qindq22::Vector{Vector{Int64}} = [[x,x,y,y]::Vector{Int64} for x=1:h11,y=1:h11 if x>y]
    quart31log1::Matrix{Float64} = zeros(Float64,size(logL,1),size(qindq31,1))
    quart22log1::Matrix{Float64} = zeros(Float64,size(logL,1),size(qindq22,1))
    quartiilog1::Matrix{Float64} = zeros(Float64,size(logL,1),h11)
    quart31sign1::Matrix{Float64} = zeros(Int64,size(logL,1),size(qindq31,1))
    quart22sign1::Matrix{Float64} = zeros(Int64,size(logL,1),size(qindq22,1))
    quartiisign1::Matrix{Float64} = zeros(Int64,size(logL,1),h11)
    quart31log::Vector{Float64} = zeros(Float64,size(qindq31,1))
    quart22log::Vector{Float64} = zeros(Float64,size(qindq22,1))
    quartdiaglog::Vector{Float64} = zeros(Float64,h11)
    quart31sign::Vector{Int} = zeros(Int,size(qindq31,1))
    quart22sign::Vector{Int} = zeros(Int,size(qindq22,1))
    quartdiagsign::Vector{Int} = zeros(Int,h11)
    @inbounds for k=1:size(qindq31,1)
        i,_,_,j = qindq31[k]
        quart31sign1[:,k] = signL .* signQMs[:,i] .* signQMs[:,i] .* signQMs[:,i] .* signQMs[:,j]
        quart31log1[:,k] = logL .+ (logQMs[:,i] + logQMs[:,i] .+ logQMs[:,i] + logQMs[:,j])
        quart31sign[k],quart31log[k] = gauss_log(quart31sign1[:,k],quart31log1[:,k])
    end
    @inbounds for k=1:size(qindq22,1)
        i,_,_,j = qindq22[k]
        quart22sign1[:,k] = signL .* signQMs[:,i] .* signQMs[:,i] .* signQMs[:,j] .* signQMs[:,j]
        quart22log1[:,k] = logL .+ (logQMs[:,i] + logQMs[:,i] .+ logQMs[:,j] + logQMs[:,j])
        quart22sign[k],quart22log[k] = gauss_log(quart22sign1[:,k],quart22log1[:,k])
    end
    @inbounds for k=1:h11
        quartiisign1[:,k] = signL .* signQMs[:,k] .* signQMs[:,k] .* signQMs[:,k] .* signQMs[:,k]
        quartiilog1[:,k] = logL .+ (logQMs[:,k] + logQMs[:,k] .+ logQMs[:,k] + logQMs[:,k])
        quartdiagsign[k],quartdiaglog[k] = gauss_log(quartiisign1[:,k],quartiilog1[:,k])
    end
    qindqdiag::Vector{Vector{Int64}} = [[x,x,x,x]::Vector{Int64} for x=1:h11]
    
    fpert::Vector{Float64} = @.(Hvals+log10(constants()["MPlanck"])- (0.5*quartdiaglog*log10(exp(1))))
    
    vals =  Hsign, Hvals .+ Float64(log10(constants()["MPlanck"])) .+9 .+ Float64(constants()["log2π"]), 
    fK .+ Float64(log10(constants()["MPlanck"])) .- Float64(constants()["log2π"]), fpert .- Float64(constants()["log2π"]), quartdiagsign, quartdiaglog .*log10(exp(1)) .+ 4*Float64(constants()["log2π"]), Array(hcat(qindq31...) .-1), quart31sign, 
    quart31log .*log10(exp(1)) .+ 4*Float64(constants()["log2π"]), quart22sign, 
    quart22log .*log10(exp(1)) .+ 4*Float64(constants()["log2π"]), Array(hcat(qindq22...) .-1)

    keys = ["msign","m", "fK", "fpert","λselfsign", "λself","λ31_i","λ31sign","λ31", "λ22_i","λ22sign","λ22"]
    return Dict(zip(keys,vals))
#     GC.gc()
end

function hp_spectrum(h11::Int,tri::Int,cy::Int=1; prec=5_000)
    pot_data = potential(h11,tri,cy);
    L::Matrix{Float64}, Q::Matrix{Int}, K::Hermitian{Float64, Matrix{Float64}} = pot_data["L"],pot_data["Q"],pot_data["K"]
    LQtest = hcat(L,Q);
    Lfull::Vector{Float64} = LQtest[:,2]
    LQsorted = LQtest[sortperm(Lfull, rev=true), :]
    Lsorted_test,Qsorted_test = LQsorted[:,1:2], Int.(LQsorted[:,3:end])
    Qtilgauss_diff = Qsorted_test[1,:]
    Ltilgauss_diff = Lsorted_test[1,:]
    for i=2:size(Qsorted_test,1)
        S = MatrixSpace(Nemo.ZZ, size(Qtilgauss_diff,1), (size(Qtilgauss_diff,2)+1))
        m = S(hcat(Qtilgauss_diff,Qsorted_test[i,:]))
        (d,bmat) = Nemo.nullspace(m)
        if d == 0
            Qtilgauss_diff = hcat(Qtilgauss_diff,Qsorted_test[i,:])
            Ltilgauss_diff = hcat(Ltilgauss_diff,Lsorted_test[i,:])
        end
    end
    spectrum_data = hp_spectrum(K,Ltilgauss_diff,Qtilgauss_diff)
end
function hp_spectrum_save(h11::Int,tri::Int,cy::Int=1)
    if h11!=0
        pot_data = potential(h11,tri,cy);
        L::Matrix{Float64}, Q::Matrix{Int}, K::Hermitian{Float64, Matrix{Float64}} = pot_data["L"],pot_data["Q"],pot_data["K"]
        LQtest = hcat(L,Q);
        Lfull::Vector{Float64} = LQtest[:,2]
        LQsorted = LQtest[sortperm(Lfull, rev=true), :]
        Lsorted_test,Qsorted_test = LQsorted[:,1:2], Int.(LQsorted[:,3:end])
        Qtilgauss_diff = Qsorted_test[1,:]
        Ltilgauss_diff = Lsorted_test[1,:]
        for i=2:size(Qsorted_test,1)
            S = MatrixSpace(Nemo.ZZ, size(Qtilgauss_diff,1), (size(Qtilgauss_diff,2)+1))
            m = S(hcat(Qtilgauss_diff,Qsorted_test[i,:]))
            (d,bmat) = Nemo.nullspace(m)
            if d == 0
                Qtilgauss_diff = hcat(Qtilgauss_diff,Qsorted_test[i,:])
                Ltilgauss_diff = hcat(Ltilgauss_diff,Lsorted_test[i,:])
            end
        end
        spectrum_data = hp_spectrum(K,Ltilgauss_diff,Qtilgauss_diff)
        h5open(cyax_file(h11,tri,cy), "r+") do file
            f2 = create_group(file, "spectrum")
            f2a = create_group(f2, "quartdiag")
            f2a["log10",gauss_diffflate=9] = spectrum_data["λself"]
            f2a["sign",gauss_diffflate=9] = spectrum_data["λselfsign"]
            f2e = create_group(f2, "gauss_diffcay")
            f2e["fpert",gauss_diffflate=9] = spectrum_data["fpert"]
            f2e["fK",gauss_diffflate=9] = spectrum_data["fK"]

            f2b = create_group(f2, "quart31")
            f2b["log10",gauss_diffflate=9] = spectrum_data["λ31"]
            f2b["sign",gauss_diffflate=9] = spectrum_data["λ31sign"]
            f2b["ingauss_diffx",gauss_diffflate=9] = spectrum_data["λ31_i"]

            f2c = create_group(f2, "quart22")
            f2c["log10",gauss_diffflate=9] = spectrum_data["λ22"]
            f2c["sign",gauss_diffflate=9] = spectrum_data["λ22sign"]
            f2c["ingauss_diffx",gauss_diffflate=9] = spectrum_data["λ22_i"]

            f2d = create_group(f2, "masses")
            f2d["log10",gauss_diffflate=9] = spectrum_data["m"]
            f2d["sign",gauss_diffflate=9] = spectrum_data["msign"]
        end
    end
    GC.gc()
end

function project_out(v::Vector{Float64})
    idd = Matrix{Float64}(I(size(v,1)))
    norm2 = dot(v,v)
    proj = 1. /norm2 * (v * v')
    return idd-proj
end

function pq_spectrum(K::Hermitian{Float64, Matrix{Float64}}, L::Matrix{Float64}, Q::Matrix{Int})
    h11::Int = size(K,1)
    fK::Vector{Float64} = log10.(sqrt.(eigen(K).values))
    Kls = cholesky(K).L
    
    LQtest::Matrix{Float64} = hcat(L,Q);
    Lfull::Vector{Float64} = LQtest[:,2]
    LQsorted::Matrix{Float64} = LQtest[sortperm(Lfull, rev=true), :]
    Lsorted_test::Matrix{Float64},Qsorted_test::Matrix{Int} = LQsorted[:,1:2], Int.(LQsorted[:,3:end])
    Qtilgauss_diff::Matrix{Int} = hcat(zeros(Int,size(Qsorted_test[1,:],1)),Qsorted_test[1,:])
    Ltilgauss_diff::Matrix{Float64} = hcat(zeros(Float64,size(Lsorted_test[1,:],1)),Lsorted_test[1,:])
    for i=2:size(Qsorted_test,1)
        S::Nemo.FmpzMatSpace = MatrixSpace(Nemo.ZZ, size(Qtilgauss_diff,1), (size(Qtilgauss_diff,2)))
        m::Nemo.fmpz_mat = S(hcat(Qtilgauss_diff[:,2:end],Qsorted_test[i,:]))
        (d::Int,_) = Nemo.nullspace(m)
        if d == 0
            Qtilgauss_diff = hcat(Qtilgauss_diff,Qsorted_test[i,:])
            Ltilgauss_diff = hcat(Ltilgauss_diff,Lsorted_test[i,:])
        end
    end
    Ltilgauss_diff = Ltilgauss_diff[:,2:end]
    Qtilgauss_diff = Qtilgauss_diff[:,2:end]
    QKs::Matrix{Float64} = zeros(Float64,h11,h11)
    fapprox::Vector{Float64} = zeros(Float64,h11)
    mapprox::Vector{Float64} = zeros(h11)
    LinearAlgebra.mul!(QKs, inv(Kls'), Qtilgauss_diff')
    QKs = QKs'
    for i=1:h11
#         println(size(QKs))
        fapprox[i] = log10(1/(2π*dot(QKs[i,:],QKs[i,:])))
        mapprox[i] = 0.5*(Ltilgauss_diff[2,i]-fapprox[i])
        proj = project_out(QKs[i,:])
        #this is the scipy.linalg.orth function written out
        u, s, vh = svd(proj,full=true)
        M, N = size(u,1), size(vh,2)
        rcond = eps() * max(M, N)
        tol = maximum(s) * rcond
        num = Int.(round(sum(s[s .> tol])))
        T = u[:, 1:num]
        QKs1 = zeros(size(QKs,1), size(T,2))
        LinearAlgebra.mul!(QKs1,QKs, T)
        QKs = copy(QKs1)
    end
    vals = [mapprox[sortperm(mapprox)] .+ 9. .+ Float64(log10(constants()["MPlanck"])), fK .+ Float64(log10(constants()["MPlanck"])) .- Float64(constants()["log2π"]), 0.5 .* fapprox[sortperm(mapprox)] .+ Float64(log10(constants()["MPlanck"]))]
    keys = ["m", "fK", "fpert"]

    return Dict(zip(keys,vals))
end

function pq_spectrum(h11::Int,tri::Int,cy::Int)
    pot_data = potential(h11,tri,cy)
    K,L,Q = pot_data["K"], pot_data["L"], pot_data["Q"]
    pq_spectrum(K, L, Q)
end

function pq_spectrum_save(h11::Int,tri::Int,cy::Int=1)
    if h11!=0
        file_open::Bool = 0
        h5open(cyax_file(h11,tri,cy), "r") do file
            if haskey(file, "spectrum")
                file_open = 1
                return nothing
            end
        end
        if file_open == 0
            pot_data = potential(h11,tri,cy);
            L::Matrix{Float64}, Q::Matrix{Int}, K::Hermitian{Float64, Matrix{Float64}} = pot_data["L"],pot_data["Q"],pot_data["K"]
            spectrum_data = pq_spectrum(K,L,Q)
            h5open(cyax_file(h11,tri,1), "r+") do file
                f2 = create_group(file, "spectrum")
                f2e = create_group(f2, "gauss_diffcay")
                f2e["fpert",gauss_diffflate=9] = spectrum_data["fpert"]
                f2e["fK",gauss_diffflate=9] = spectrum_data["fK"]

                f2d = create_group(f2, "masses")
                f2d["log10",gauss_diffflate=9] = spectrum_data["m"]
            end
        end
    end
end

function Base.convert(::Type{Matrix{Int}}, x::Nemo.fmpz_mat)
    m,n = size(x)
    mat = Int[x[i,j] for i = 1:m, j = 1:n]
    return mat
end
Base.convert(::Type{Matrix}, x::Nemo.fmpz_mat) = convert(Matrix{Int}, x)


"""
    vacua(L,Q)

Compute the number of vacua given an instanton charge matrix `Q` and 2-column matrix of instanton scales `L` (in the form [sign; exponent])

For small systems (Nax<=50) the algorithm computes the ratio of volumes of the fundamental domain of the leading potential and the full potential.

For larger systems, the algorithm only computes the volume of the fundamental domain of the leading potential.
#Examples
```julia-repl
julia> using CYAxiverse
julia> h11,tri,cy = 10,20,1;
julia> pot_data = CYAxiverse.read.potential(h11,tri,cy);
julia> vacua_data = CYAxiverse.generate.vacua(pot_data["L"],pot_data["Q"])
Dict{String, Any} with 3 entries:
  "θ∥"     => Rational[1//1 0//1 … 0//1 0//1; 0//1 1//1 … 0//1 0//1; … ; 0//1 0//1 … 1//1 0//1; 0//1 0//1 … 0//1 1//1]
  "vacua"  => 3
  "Qtilgauss_diff" => [0 0 … 1 0; 0 0 … 0 0; … ; 1 1 … 0 0; 0 0 … 0 0]
```
"""
function vacua(L::Matrix{Float64},Q::Matrix{Int})
    h11::Int = size(Q,2)
    if h11<=50
        T::Nemo.fmpz_mat = snf_with_transform(matrix(Nemo.ZZ,Q))[2]
        # println(size(T))
        Tparallel1::Nemo.fmpz_mat = inv(T)[:,1:h11]
        Tparallel::Matrix{Int} = convert(Matrix{Int},Tparallel1)
        θparalleltest::Matrix{Float64} = inv(transpose(Float64.(Q)) * Float64.(Q)) * transpose(Float64.(Q)) * Float64.(Tparallel)
    end
    LQtest::Matrix{Float64} = hcat(L,Float64.(Q))
    LQsorted::Matrix{Float64} = LQtest[sortperm(L[:,2], rev=true), :]
    Lsorted_test::Matrix{Float64},Qsorted_test::Matrix{Int} = LQsorted[:,1:2], Int.(LQsorted[:,3:end])
    Qtilgauss_diff::Matrix{Int} = hcat(zeros(Int,size(Qsorted_test[1,:],1)),Qsorted_test[1,:])
    S::Nemo.FmpzMatSpace = MatrixSpace(Nemo.ZZ,1,1)
    m::Nemo.fmpz_mat = matrix(Nemo.ZZ,zeros(1,1))
    d::Int = 1
    for i=2:size(Qsorted_test,1)
        S = MatrixSpace(Nemo.ZZ, size(Qtilgauss_diff,1), (size(Qtilgauss_diff,2)))
        m = S(hcat(Qtilgauss_diff[:,2:end],Qsorted_test[i,:]))
        d = Nemo.nullspace(m)[1]
        if d == 0
            Qtilgauss_diff = hcat(Qtilgauss_diff,Qsorted_test[i,:])
        end
    end
    Qtilgauss_diff = Qtilgauss_diff[:,2:end]

    if h11 <= 50
        vacua = Int(round(abs(gauss_difft(θparalleltest) / gauss_difft(inv(Qtilgauss_diff)))))
        thparallel::Matrix{Rational} = Rational.(round.(θparalleltest; digits=5))
        keys = ["vacua","θ∥","Qtilgauss_diff"]
        vals = [abs(vacua), thparallel, Qtilgauss_diff]
        return Dict(zip(keys,vals))
    else
        vacua = Int(round(abs(1 / gauss_difft(inv(Qtilgauss_diff)))))
        keys = ["vacua","Qtilgauss_diff"]
        vals = [abs(vacua), Qtilgauss_diff]
        return Dict(zip(keys,vals))
    end
end
"""
    vacua_TB(L,Q)

Compute the number of vacua given an instanton charge matrix `Q` and 2-column matrix of instanton scales `L` (in the form [sign; exponent])

For small systems (Nax<=50) the algorithm computes the ratio of volumes of the fundamental domain of the leading potential and the full potential.

For larger systems, the algorithm only computes the volume of the fundamental domain of the leading potential.
#Examples
```julia-repl
julia> using CYAxiverse
julia> h11,tri,cy = 10,20,1;
julia> pot_data = CYAxiverse.read.potential(h11,tri,cy);
julia> vacua_data = CYAxiverse.generate.vacua_TB(pot_data["L"],pot_data["Q"])
Dict{String, Any} with 3 entries:
  "θ∥"     => Rational[1//1 0//1 … 0//1 0//1; 0//1 1//1 … 0//1 0//1; … ; 0//1 0//1 … 1//1 0//1; 0//1 0//1 … 0//1 1//1]
  "vacua"  => 11552.0
  "Qtilgauss_diff" => [0 0 … 0 1; 0 0 … 0 0; … ; 1 1 … -1 -1; 0 0 … 0 0]
```
"""
function vacua_TB(L::Matrix{Float64},Q::Matrix{Int})
    
    h11::Int = size(Q,2)
    if h11 <= 50
        ###### Nemo SNF #####
        Qtemp::Nemo.fmpz_mat = matrix(Nemo.ZZ,Q)
        T::Nemo.fmpz_mat = snf_with_transform(Qtemp)[2]
        Tparallel1::Nemo.fmpz_mat = inv(T)[:,1:h11]
        Tparallel::Matrix{Int} = convert(Matrix{Int},Tparallel1)

        ###### wildart SNF #####
        # F = smith(Q)
        # T::Matrix{Int} = F.S
        # Tparallel::Matrix{Int} = round.(inv(T)[:,1:h11])
        # println(size(T))
        
        θparalleltest::Matrix{Float64} = inv(transpose(Float64.(Q)) * Float64.(Q)) * transpose(Float64.(Q)) * Float64.(Tparallel)
    end
    LQtest::Matrix{Float64} = hcat(L,Q);
    LQsorted::Matrix{Float64} = LQtest[sortperm(L[:,2], rev=true), :]
    Lsorted_test::Matrix{Float64},Qsorted_test::Matrix{Int} = LQsorted[:,1:2], Int.(LQsorted[:,3:end])
    Qtilgauss_diff::Matrix{Int} = hcat(zeros(Int,size(Qsorted_test[1,:],1)),Qsorted_test[1,:])
    Ltilgauss_diff::Matrix{Float64} = hcat(zeros(Float64,size(Lsorted_test[1,:],1)),Lsorted_test[1,:])
    S::Nemo.FmpzMatSpace = MatrixSpace(Nemo.ZZ,1,1)
    m::Nemo.fmpz_mat = matrix(Nemo.ZZ,zeros(1,1))
    d::Int = 1
    Qbar::Matrix{Int} = zeros(Int,size(Qsorted_test[1,:],1),1)
    Lbar::Matrix{Float64} = zeros(Float64,size(Lsorted_test[1,:],1),1)
    for i=2:size(Qsorted_test,1)
        S = MatrixSpace(Nemo.ZZ, size(Qtilgauss_diff,1), (size(Qtilgauss_diff,2)))
        m = S(hcat(Qtilgauss_diff[:,2:end],Qsorted_test[i,:]))
        d = Nemo.nullspace(m)[1]
        if d == 0
            Qtilgauss_diff = hcat(Qtilgauss_diff,Qsorted_test[i,:])
            Ltilgauss_diff = hcat(Ltilgauss_diff,Lsorted_test[i,:])
    else
        Qbar = hcat(Qbar,Qsorted_test[i,:])
        Lbar = hcat(Lbar,Lsorted_test[i,:])
        end
    end
    Qtilgauss_diff = Qtilgauss_diff[:,2:end]
    Qbar = Qbar[:,2:end]
    Ltilgauss_diff = Ltilgauss_diff[:,2:end]
    Lbar = Lbar[:,2:end]
    Ltilgauss_diff_min::Float64 = minimum(Ltilgauss_diff[2,:])
    Ldiff_limit::Float64 = log10(0.5)
    Qbar = Qbar[:, Lbar[2,:] .>= (Ltilgauss_diff_min + Ldiff_limit)]
    Lbar = Lbar[:,Lbar[2,:] .>= (Ltilgauss_diff_min + Ldiff_limit)]
    α::Matrix{Float64} = round.(Qbar' * inv(Qtilgauss_diff');digits=5)
    for i=1:size(α,1)
        ingauss_diffx=0
        for j=1:size(α,2)
            if α[i,j] != 0.
                ingauss_diffx = j
            end
        end
        if ingauss_diffx!=0
            Ldiff::Float64 = round(Lbar[2,i] - Ltilgauss_diff[2,ingauss_diffx], digits=3)
            if Ldiff > Ldiff_limit
                Qtilgauss_diff = hcat(Qtilgauss_diff,Qbar[:,i])
                Ltilgauss_diff = hcat(Ltilgauss_diff,Lbar[:,i]) 
            end
        end
    end
    if h11 <= 50
        if size(Qtilgauss_diff,1) == size(Qtilgauss_diff,2)
            vacua = abs(gauss_difft(θparalleltest) / gauss_difft(inv(Qtilgauss_diff)))
        else
            vacua = abs(gauss_difft(θparalleltest) / (1/sqrt(gauss_difft(Qtilgauss_diff * Qtilgauss_diff'))))
        end
        thparallel::Matrix{Rational} = Rational.(round.(θparalleltest; digits=5))
        keys = ["vacua","θ∥","Qtilgauss_diff"]
        vals = [abs(vacua), thparallel, Qtilgauss_diff]
        return Dict(zip(keys,vals))
    else
        if size(Qtilgauss_diff,1) == size(Qtilgauss_diff,2)
            vacua = abs(1 / gauss_difft(inv(Qtilgauss_diff)))
        else
            vacua = abs(sqrt(gauss_difft(Qtilgauss_diff * Qtilgauss_diff')))
        end
        
        keys = ["vacua","Qtilgauss_diff"]
        vals = [abs(vacua), Qtilgauss_diff]
        return Dict(zip(keys,vals))
    end
end

"""
    vacua_TB(h11,tri,cy)

Compute the number of vacua given a geometry from the KS database.

For small systems (Nax<=50) the algorithm computes the ratio of volumes of the fundamental domain of the leading potential and the full potential.

For larger systems, the algorithm only computes the volume of the fundamental domain of the leading potential.
#Examples
```julia-repl
julia> using CYAxiverse
julia> h11,tri,cy = 10,20,1;
julia> vacua_data = CYAxiverse.generate.vacua_TB(h11,tri,cy)
Dict{String, Any} with 3 entries:
  "θ∥"     => Rational[1//1 0//1 … 0//1 0//1; 0//1 1//1 … 0//1 0//1; … ; 0//1 0//1 … 1//1 0//1; 0//1 0//1 … 0//1 1//1]
  "vacua"  => 11552.0
  "Qtilgauss_diff" => [0 0 … 0 1; 0 0 … 0 0; … ; 1 1 … -1 -1; 0 0 … 0 0]
```
"""
function vacua_TB(h11::Int,tri::Int,cy::Int)
    pot_data = potential(h11,tri,cy)
    Q::Matrix{Int}, L::Matrix{Float64} = pot_data["Q"], pot_data["L"] 
    vacua_TB(L,Q)
end


function vacua_save(h11::Int,tri::Int,cy::Int=1)
    file_open::Bool = 0
    h5open(cyax_file(h11,tri,cy), "r") do file
        if haskey(file, "vacua")
            file_open = 1
            return nothing
        end
    end
    if file_open == 0
        pot_data = potential(h11,tri,cy)
        vacua_data = vacua(pot_data["L"],pot_data["Q"])
        h5open(cyax_file(h11,tri,cy), "r+") do file
            f3 = create_group(file, "vacua")
            f3["vacua",gauss_diffflate=9] = vacua_data["vacua"]
            f3["Qtilgauss_diff",gauss_diffflate=9] = vacua_data["Qtilgauss_diff"]
            if h11 <=50
                f3a = create_group(f3, "thparallel")
                f3a["numerator",gauss_diffflate=9] = numerator.(vacua_data["θ∥"])
                f3a["gauss_diffnominator",gauss_diffflate=9] = gauss_diffnominator.(vacua_data["θ∥"])
            end
        end
    end
end



function vacua_save_TB(h11::Int,tri::Int,cy::Int=1)
    file_open::Bool = 0
    h5open(cyax_file(h11,tri,cy), "r") do file
        if haskey(file, "vacua_TB")
            file_open = 1
            return nothing
        end
    end
    if file_open == 0
        pot_data = potential(h11,tri,cy)
        vacua_data = vacua_TB(pot_data["L"],pot_data["Q"])
        h5open(cyax_file(h11,tri,cy), "r+") do file
            f3 = create_group(file, "vacua_TB")
            f3["vacua",gauss_diffflate=9] = vacua_data["vacua"]
            f3["Qtilgauss_diff",gauss_diffflate=9] = vacua_data["Qtilgauss_diff"]
            if h11 <=50
                f3a = create_group(f3, "thparallel")
                f3a["numerator",gauss_diffflate=9] = numerator.(vacua_data["θ∥"])
                f3a["gauss_diffnominator",gauss_diffflate=9] = gauss_diffnominator.(vacua_data["θ∥"])
            end
        end
    end
end
end