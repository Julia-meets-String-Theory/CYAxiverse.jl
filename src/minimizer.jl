module minimizer

using HDF5
using LinearAlgebra
using ArbNumerics, Tullio, LoopVectorization
using GenericLinearAlgebra
using Distributions
using Optim, LineSearches, Dates, HDF5

using ..filestructure: cyax_file, minfile, present_dir
using ..read: potential

function minimize(h11::Int,tri::Int,cy::Int,LV::Vector,QV::Matrix,x0::Vector,gradσ::Matrix,θparalleltest::Matrix,Qtilde::Matrix,algo,prec)
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

        keys = ["±V", "logV","±x", "logx", "±a","loga", "±ã", "logã"]
        vals = [Vmin_sign, Vmin_log, xmin_sign, xmin_log, a_sign, a_log, atilde_sign, atilde_log]
        return Dict(zip(keys,vals))
        GC.gc()
    end
end

function minimize_save(h11::Int,tri::Int,cy::Int,LV::Vector,QV::Matrix,x0::Vector,gradσ::Matrix,θparalleltest::Matrix,Qtilde::Matrix,algo; prec::Int=5_000, run_num::Int=1)
    min_data = minimize(h11,tri,cy,LV,QV,x0,gradσ,θparalleltest,Qtilde,algo, prec)
    if min_data == nothing
        return nothing
    else
        h5open(CYAxiverse.filestructure.minfile(h11,tri,cy),isfile(CYAxiverse.filestructure.minfile(h11,tri,cy)) ? "r+" : "w") do file
            if haskey(file, "runs")
            else
                f0 = create_group(file,"runs")
            end
            f0 = create_group(file, "runs/$run_num")
            f1 = create_group(f0, "V")
            f1["log10",deflate=9] = min_data["logV"]
            f1["sign",deflate=9] = min_data["±V"]
            f2 = create_group(f0, "x")
            f2["log10",deflate=9] = min_data["logx"]
            f2["sign",deflate=9] = min_data["±x"]
            f3 = create_group(f0, "a")
            f3["log10",deflate=9] = min_data["loga"]
            f3["sign",deflate=9] = min_data["±a"]
            f4 = create_group(f0, "atilde")
            f4["log10",deflate=9] = min_data["logã"]
            f4["sign",deflate=9] = min_data["±ã"]
        end
    end
GC.gc()
end
function grad_std(h11::Int,tri::Int,cy::Int,LV::Vector,QV::Matrix)
    Arb0 = ArbFloat(0.)
    Arb1 = ArbFloat(1.)
    Arb2π = ArbFloat(2π)
    function QX(x::Vector)
        Qx = zeros(ArbFloat,size(QV,1));
        @tullio Qx[c] = QV[c,i] * x[i]
        return Qx
    end
    grad(x) = vcat([dot(LV,QV[:,i] .* sin.(QX(x))) for i ∈ 1:h11]...)
    n=100
    grad_all = zeros(h11,n)
    for j=1:n
        x0 = ArbFloat.(rand(Uniform(0,2π),h11)) .* rand(ArbFloat,h11)
        grad_all[:,j] = grad(x0)
    end
    return ArbFloat.(std(grad_all, dims=2))
end

end
