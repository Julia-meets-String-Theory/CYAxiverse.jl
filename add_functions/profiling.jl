module profiling

using HDF5
using LinearAlgebra
using ArbNumerics, Tullio, LoopVectorization, Nemo
using GenericLinearAlgebra
using Distributions
using TimerOutputs
using SmithNormalForm

using ..filestructure: cyax_file, minfile, present_dir
using ..read: potential

function vacua(L::Matrix{Float64},Q::Matrix{Int})
    reset_timer!()
    @timeit "h11" h11::Int = size(Q,2)
    if h11 < 50
        ###### Nemo SNF #####
        @timeit "Nemo matrix" Qtemp::Nemo.fmpz_mat = matrix(Nemo.ZZ,Q)
        @timeit "SNF" T::Nemo.fmpz_mat = snf_with_transform(Qtemp)[2]
        @timeit "inv(T)" Tparallel1::Nemo.fmpz_mat = inv(T)[:,1:h11]
        @timeit "convert T∥" Tparallel::Matrix{Int} = convert(Matrix{Int},Tparallel1)

        ###### wildart SNF #####
        # @timeit "SNF" F = smith(Q)
        # @timeit "T" T::Matrix{Int} = F.S
        # @timeit "inv(T)" Tparallel::Matrix{Int} = round.(inv(T)[:,1:h11])
        # println(size(T))
        
        @timeit "θparallel" θparalleltest::Matrix{Float64} = inv(transpose(Float64.(Q)) * Float64.(Q)) * transpose(Float64.(Q)) * Float64.(Tparallel)
    end
    @timeit "zip LQ" LQtest::Matrix{Float64} = hcat(L,Q);
    @timeit "sort LQ" LQsorted::Matrix{Float64} = LQtest[sortperm(L[:,2], rev=true), :]
    @timeit "unzip LQ" Lsorted_test::Matrix{Float64},Qsorted_test::Matrix{Int} = LQsorted[:,1:2], Int.(LQsorted[:,3:end])
    @timeit "init Qtilde" Qtilde::Matrix{Int} = hcat(zeros(Int,size(Qsorted_test[1,:],1)),Qsorted_test[1,:])
    @timeit "init Ltilde" Ltilde::Matrix{Float64} = hcat(zeros(Float64,size(Lsorted_test[1,:],1)),Lsorted_test[1,:])
    @timeit "init S" S::Nemo.FmpzMatSpace = MatrixSpace(Nemo.ZZ,1,1)
    @timeit "init m" m::Nemo.fmpz_mat = matrix(Nemo.ZZ,zeros(1,1))
    d::Int = 1
    @timeit "init Qbar" Qbar::Matrix{Int} = zeros(Int,size(Qsorted_test[1,:],1),1)
    @timeit "init Lbar" Lbar::Matrix{Float64} = zeros(Float64,size(Lsorted_test[1,:],1),1)
    for i=2:size(Qsorted_test,1)
        @timeit "Matrix.Space" S = MatrixSpace(Nemo.ZZ, size(Qtilde,1), (size(Qtilde,2)))
        @timeit "lin. ind." m = S(hcat(Qtilde[:,2:end],Qsorted_test[i,:]))
        @timeit "NullSpace" d = Nemo.nullspace(m)[1]
        if d == 0
            @timeit "Qtilde" Qtilde = hcat(Qtilde,Qsorted_test[i,:])
            @timeit "Ltilde" Ltilde = hcat(Ltilde,Lsorted_test[i,:])
    else
        @timeit "Qbar" Qbar = hcat(Qbar,Qsorted_test[i,:])
        @timeit "Lbar" Lbar = hcat(Lbar,Lsorted_test[i,:])
        end
    end
    @timeit "Qtilde first pass" Qtilde = Qtilde[:,2:end]
    @timeit "Qbar first pass" Qbar = Qbar[:,2:end]
    @timeit "Ltilde first pass" Ltilde = Ltilde[:,2:end]
    @timeit "Lbar first pass" Lbar = Lbar[:,2:end]
    println(size(Qbar), size(Lbar),size(Ltilde),size(Qtilde))
    @timeit "Ltilde min" Ltilde_min::Float64 = minimum(Ltilde[2,:])
    println(Ltilde_min)
    @timeit "Ldiff limit" Ldiff_limit::Float64 = log10(0.01)
    @timeit "Qbar reduce" Qbar = Qbar[:, Lbar[2,:] .>= (Ltilde_min + Ldiff_limit)]
    @timeit "Lbar reduce" Lbar = Lbar[:,Lbar[2,:] .>= (Ltilde_min + Ldiff_limit)]
    @timeit "alpha" α::Matrix{Float64} = round.(Qbar' * inv(Qtilde'))
    println(size(Qbar), size(Lbar), size(α), Qbar[:,1], Lbar[1,2])
    # println(α)
    # i=1
    for i=1:size(α,1)
        index=0
        for j=1:size(α,2)
            if α[i,j] != 0.
                index = j
            end
        end
        if index!=0
            Ldiff::Float64 = round(Lbar[2,i] - Ltilde[2,index], digits=3)
            if Ldiff > Ldiff_limit
                println([i index α[i,index] Ldiff Lbar[2,i] Ltilde[2,index]])
                @timeit "Qtilde 2nd pass" Qtilde = hcat(Qtilde,Qbar[:,i])
                @timeit "Ltilde 2nd pass" Ltilde = hcat(Ltilde,Lbar[:,i]) 
            end
        end
    end
    println(size(Qtilde))
    if h11 < 50
        if size(Qtilde,1) == size(Qtilde,2)
            @timeit "vacua square" vacua = Int(round(abs(det(θparalleltest) / det(inv(Qtilde)))))
        else
            @timeit "vacua P!=N" vacua = round(abs(det(θparalleltest) / (1/sqrt(det(Qtilde * Qtilde')))))
        end
        thparallel::Matrix{Rational} = Rational.(round.(θparalleltest; digits=5))
        keys = ["vacua","θ∥","Qtilde"]
        vals = [abs(vacua), thparallel, Qtilde]
        print_timer()
        return Dict(zip(keys,vals))
    else
        if size(Qtilde,1) == size(Qtilde,2)
            @timeit "vacua square" vacua = Int(round(abs(1 / det(inv(Qtilde)))))
        else
            @timeit "vacua P!=N" vacua = round(abs(sqrt(det(Qtilde * Qtilde'))))
        end
        
        keys = ["vacua","Qtilde"]
        vals = [abs(vacua), Qtilde]
        print_timer()
        return Dict(zip(keys,vals))
    end
end


function minimiser(h11::Int,tri::Int,cy::Int,LV::Vector,QV::Matrix,x0::Vector,gradσ::Matrix,θparalleltest::Matrix,Qtilde::Matrix,algo,prec)
    setprecision(ArbFloat,digits=prec)
    Arb0 = ArbFloat(0.)
    Arb1 = ArbFloat(1.)
    Arb2π = ArbFloat(2π)
    threshold = 0.01
    function QX(x::Vector)
        Qx = zeros(ArbFloat,size(QV,1));
        @tullio Qx[c] = QV[c,i] * x[i]
        return Qx
    end
    function fitness(x::Vector)
        V = dot(LV,(Arb1 .- cos.(QX(x))))
        return V
    end
    function grad!(gradient::Vector, x::Vector)
        grad_temp = zeros(ArbFloat, size(LV,1),h11)
        @tullio grad_temp[c,i] = QV[c,i] * sin.(QX(x)[c])
        @tullio gradient[i] = LV[c] * grad_temp[c,i]
    end
    function hess(x::Vector)
        grad2::Matrix{ArbFloat} = zeros(ArbFloat,(h11,h11))
        hind1::Vector{Vector{Int64}} = [[x,y]::Vector{Int64} for x=1:h11,y=1:h11 if x>=y]
        grad2_temp::Vector{ArbFloat} = zeros(ArbFloat,size(hind1,1))
        grad2_temp1::Matrix{Float64} = zeros(Float64,size(LV,1),size(hind1,1))
        @tullio grad2_temp1[c,k] = @inbounds(begin
        i,j = hind1[k]
                QV[c,i] * QV[c,j] * cos.(QX(x)[c]) end) grad=false
        @tullio grad2_temp[k] = grad2_temp1[c,k] * LV[c]
        @inbounds for i=1:size(hind1,1)
            j,k = hind1[i]
            grad2[j,k] = grad2_temp[i]
        end
        hessfull = Hermitian(grad2 + transpose(grad2) - Diagonal(grad2))
    end
    function hess!(hessian::Matrix, x::Vector)
        grad2 = zeros(ArbFloat,(h11,h11))
        hind1 = [[x,y]::Vector{Int64} for x=1:h11,y=1:h11 if x>=y]
        grad2_temp = zeros(ArbFloat,size(hind1,1))
        grad2_temp1 = zeros(ArbFloat,size(LV,1),size(hind1,1))
        @tullio grad2_temp1[c,k] = @inbounds(begin
                i,j = hind1[k]
                QV[c,i] * QV[c,j] * cos.(QX(x)[c]) end) grad=false avx=false
        @tullio grad2_temp[k] = grad2_temp1[c,k] * LV[c]
        @inbounds for i=1:size(hind1,1)
            j,k = hind1[i]
            grad2[j,k] = grad2_temp[i]
        end
        hessian .= grad2 + transpose(grad2) - Diagonal(grad2)
    end
    grad(x) = vcat([dot(LV,QV[:,i] .* sin.(QX(x))) for i ∈ 1:h11]...)
    res = optimize(fitness,grad!,hess!,
                x0, algo,
                Optim.Options(x_tol =minimum(abs.(LV)),g_tol =minimum(threshold .* abs.(gradσ))))
    Vmin = Optim.minimum(res)
    xmin = Optim.minimizer(res)
    GC.gc()
    if Float64(log10(abs(minimum(eigen(hess(xmin)).values)))) < -prec && sum(Float64.(log10.(abs.(grad(xmin)))) .< log10.(abs.(threshold .* gradσ))) == (h11 - size(gradσ[gradσ .== 0.],1))
        a = mod.(((ArbFloat.(θparalleltest) * xmin)/Arb2π),Arb1)
        atilde = ArbFloat.(Qtilde) * xmin/Arb2π
        a_sign = Int.(sign.(a))
        a_log = Float64.(log10.(abs.(a)))
        atilde_sign = Int.(sign.(atilde))
        atilde_log = Float64.(log10.(abs.(atilde)))
        Vmin_sign = Int(sign(Vmin))
        Vmin_log = Float64(log10(abs(Vmin)))
        xmin_log = Float64.(log10.(abs.(xmin)))
        xmin_sign = Int.(sign.(xmin))

        keys = ["±V", "logV","±x", "logx", "±a","loga", "±ã", "logã", "eig(H)", "grad"]
        vals = [Vmin_sign, Vmin_log, xmin_sign, xmin_log, a_sign, a_log, atilde_sign, atilde_log,Float64(log10(abs(minimum(eigen(hess(xmin)).values)))),sum(Float64.(log10.(abs.(grad(xmin)))) .< log10.(abs.(threshold .* gradσ)))]
        return Dict(zip(keys,vals))
        GC.gc()
    end
end

end