### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 3788df6d-c756-4b6a-8d75-8cd018ab2991
begin
	import Pkg	
	Pkg.activate("/scratch/users/mehta2/cyaxiverse/CYAxiverse")
end

# ╔═╡ d1f78454-3ba0-4569-a491-f52e737c7dc3
begin
	using Revise	
	using HDF5, ArbNumerics, LineSearches, Optim, CairoMakie, Distributions, LinearAlgebra, ProgressLogging, Nemo
	using CYAxiverse
end

# ╔═╡ 7fb60b52-1158-4bcb-b5e0-1e3ebd1ad52b
using PlutoUI

# ╔═╡ fee399f9-2668-41e0-a296-37b348a04769
md"""
# Vacua search -- Numerics
"""

# ╔═╡ 90f44877-6310-49b1-9331-f8601918e4b3
md"""
### This notebook randomly selects a geometry in our database that may contain multiple minima and then runs a numerical optimisation routine to find them
"""

# ╔═╡ 2000a078-38f5-4c93-8627-ba6b4970aef6
html"""
<style>
	main {
		margin: 0 auto;
		max-width: 95%;
    	padding-left: max(160px, 10%);
    	padding-right: max(160px, 10%);
	}
</style>
"""

# ╔═╡ 7c8e7502-94d8-4da6-a5e2-b950b33a62c2
begin
	ENV["newARGS"] = string("vacua_stretch")
	md"""
	The database we are using is $(ENV["newARGS"])
	"""
end

# ╔═╡ 9071d014-a286-4f6f-bafc-648c1954d3d7
begin
	h11list = CYAxiverse.filestructure.paths_cy()[2]
	md"""
	There are a total of $(size(h11list,2)) geometries in the database with $(minimum(Set(h11list[1,:]))) ≤ ``h^{1,1}`` ≤ $(maximum(Set(h11list[1,:])))
	"""
end

# ╔═╡ 25e1eb34-0e4b-4bf2-ac1b-228545bb82a8
h11list

# ╔═╡ 4c1b071d-65ad-41ce-b7ab-9b11fcf15ce5
@bind go PlutoUI.Button("Run another example")

# ╔═╡ e1fdc533-efc0-4012-8464-3db601f66819
begin
	min_idx = 7_000
	max_idx = 35_001
	md"""
	Finding a suitable example between $(h11list[1,min_idx]) `` \leq h^{1,1} < `` $(h11list[1,max_idx])
	"""
end

# ╔═╡ 8699f77c-4a59-4194-9f08-f878b65b93ba
h11, tri, cy = 35, 10, 1

# ╔═╡ 281b0be2-86fd-4201-871a-7e362b5872d7
# ╠═╡ disabled = true
#=╠═╡
# let 
# 	go
# 	@progress for i=rand(min_idx:max_idx,1000)
# 			global h11,tri,cy = h11list[:,i]
# 			LQtildebar_data = CYAxiverse.generate.LQtildebar(h11,tri,cy; threshold=0.01)
# 			Qtilde = LQtildebar_data["Qtilde"]
# 			if size(Qtilde,1) != size(Qtilde,2) && size(LQtildebar_data["Qeff"],1) ≤ 20
# 				break
# 			else
# 				global h11, tri, cy = 9, 256, 1
# 			end
# 	end;
# end;
  ╠═╡ =#

# ╔═╡ 871778ce-735d-49ed-b6f5-ab68f127ad32
md"""
The system we are considering is identified by:
	``h^{1,1}`` = $h11; Polytope number = $tri; FRST number = $cy
"""

# ╔═╡ 0b13b99a-9c7d-4117-90ba-e74baf49054d
pot_data = CYAxiverse.read.potential(h11,tri,cy);

# ╔═╡ 4cfe15ea-1a7f-4af3-9602-6eac3732232a
τ = CYAxiverse.read.geometry(h11,tri,cy)["τ_volumes"]

# ╔═╡ 0b261480-4109-41f9-ac4e-52d040b20efe
Kinv = CYAxiverse.read.geometry(h11,tri,cy)["Kinv"]

# ╔═╡ 33fc2f88-c402-4394-83c9-10df2e7a6d96
qprime = pot_data["Q"][1:h11+4,:]

# ╔═╡ 6c6bd3c6-4019-426c-9188-e63debe0fe8b
size(pot_data["L"])

# ╔═╡ b5020c4c-d37f-4ad0-9633-ebdef6898afc
rhs_constraint = zeros(size(qprime,1));

# ╔═╡ 1c45d205-f8fb-4d1f-8d57-0d6d46ce622f
lhs_constraint = zeros(size(qprime,1),size(qprime,1));

# ╔═╡ fd53a8a1-c782-4f31-bf8a-1403de56b64d
for i=1:h11+4
	for j=1:h11+4
		if i>j
			lhs_constraint[i,j] = log.(abs.(pi*dot(qprime[i,:],(Kinv * qprime[j,:])))) .+ (-2π * dot(τ, qprime[i,:] .+ qprime[j,:]))
		end
	end
	rhs_constraint[i] = log.(abs.(dot(qprime[i,:],τ))) .+ (-2π * dot(τ, qprime[i,:]))
end

# ╔═╡ 4993e5d8-48f1-4b4e-878c-9576e7bfc85c
lhs_constraint

# ╔═╡ 5abd2caa-8d2d-4c8b-b064-a0426447b392
LowerTriangular(lhs_constraint .> rhs_constraint) - I(h11+4) == LowerTriangular(zeros(h11+4, h11+4))

# ╔═╡ 07f43035-2e72-47b1-baef-7258c65a13c8
constraint = LowerTriangular(lhs_constraint .> rhs_constraint) - I(h11+4)

# ╔═╡ 64f3ac82-0f2b-4092-b904-b750ec16a309
[i for (i,item) in enumerate(eachrow(constraint)) if sum(item .== 0) != h11+4]

# ╔═╡ 6f614ec4-c03f-4d65-bd3b-153a31c23da4
[rhs_constraint[i] for i in [i for (i,item) in enumerate(eachrow(constraint)) if sum(item .== 0) != h11+4]]

# ╔═╡ 622570d7-a3fa-4bd0-8d1d-5762490e8ad0
lhs_constraint[[i for (i,item) in enumerate(eachrow(constraint)) if sum(item .== 0) != h11+4], [i for (i,item) in enumerate(eachcol(constraint)) if sum(item .== 0) != h11+4]]

# ╔═╡ 6c5a1870-3657-4978-995f-578fe9d0602c
lhs_constraint[[i for (i,item) in enumerate(eachrow(constraint)) if sum(item .== 0) != h11+4], :]

# ╔═╡ cd6bbbd5-321f-4868-b0e7-71b96965bef7
constraint[[i for (i,item) in enumerate(eachrow(constraint)) if sum(item .== 0) != h11+4],[i for (i,item) in enumerate(eachcol(constraint)) if sum(item .== 0) != h11+4]]

# ╔═╡ 6f9c2ba1-6536-44d2-938e-7adf68b93bc5


# ╔═╡ a1dfa08c-1056-4e15-a040-4de1276f400c
mod10(x) = (mod(x / 2π, 1) ≈ 1 || mod(x / 2π, 1) ≈ 0 ? 0 : x)

# ╔═╡ ecfc017c-5eca-41b2-ba3e-71d6dbe68403
function vacua_MK(L::Matrix{Float64}, Q::Matrix{Int}; threshold::Float64 = 1e-2)
	setprecision(ArbFloat,digits=5_000)
    LQtildebar = LQtildebar(L, Q; threshold=threshold)
	Ltilde = LQtildebar["Ltilde"][:,sortperm(LQtildebar["Ltilde"][2,:], rev=true)]
    Qtilde = LQtildebar["Qtilde"]'[sortperm(Ltilde[2,:], rev=true), :]
	Qtilde = Matrix{Int}(Qtilde')
    basis_vectors = zeros(size(Qtilde,1), size(Qtilde,1))
	idx = 1
    while idx ≤ size(Qtilde,2)
		Qsub = Qtilde[:,idx]
		Lsub = Ltilde[:,idx]
		while Ltilde[2, idx+1] - Ltilde[2, idx] ≥ threshold && dot(Qtilde[:,idx+1], Qtilde[:,idx]) != 0
			Lsub = hcat(Lsub, Ltilde[:, idx+1])
			Qsub = hcat(Qsub, Qtilde[:, idx+1])
			idx += 1
		end
		if size(Qsub,2) == 1
			basis_vectors[idx,:] = Qsub
			idx += 1
		else
			Lsubdiff = Lsub[2,:] .- Lsub[2,1]
			Lfull = Lsubdiff[1,:] .* 10. .^ Lsubdiff[2,:];
			Qsubmask = [sum(i .== 0) < size(Qsub,1) for i in eachcol(Qsub)]
			Qsub = Qsub[:,Qsubmask]
			for run_number = 1:10_000
				x0 = rand(Uniform(0,2π),h11) .* rand(Float64,h11)
				res = CYAxiverse.minimizer.minimize(Lfull, Qsub, x0) ##need to write subsystem minimizer
				res["Vmin_log"] = res["Vmin_log"] .+ Lsub[2,1]
			end
			xmin = hcat(res["xmin"]...)
			for i in eachcol(xmin)
				i[:] = @. ifelse(mod(i / 2π, 1) ≈ 1 || mod(i / 2π, 1) ≈ 0 ? 0 : i)
			end
			xmin = xmin[:, [sum(i)!=0 for i in eachcol(xmin)]]
			xmin = xmin[:,sortperm([sqrt(sum(abs2,i)) for i in eachcol(xmin)])]
			lattice_vecs = lattice_minimize(xmin) ##need to write lattice minimizer
			basis_vectors[idx-size(lattice_vecs,2):idx, :] = lattice_vecs
		end
        proj = project_out(Qtilde[i,:])
        #this is the scipy.linalg.orth function written out
        u, s, vh = svd(proj,full=true)
        M, N = size(u,1), size(vh,2)
        rcond = eps() * max(M, N)
        tol = maximum(s) * rcond
        num = Int.(round(sum(s[s .> tol])))
        T = u[:, 1:num]
        Qtilde_i = zeros(size(Qtilde, 1), size(T, 2))
        LinearAlgebra.mul!(Qtilde_i, Qtilde, T)
        Qtilde = copy(Qtilde_i)
    end
    return basis_vectors
end

# ╔═╡ dc4c35f6-90cc-4a71-ab8d-b9ae847ca561
size(log.(abs.([(pi*dot(qprime[i,:],(Kinv * qprime[j,:]))) for i=1:h11+4,j=1:h11+4 if i<j] )) .+ [(-2π * dot(τ, qprime[i,:] .+ qprime[j,:])) for i=1:h11+4,j=1:h11+4 if i<j])

# ╔═╡ ac863bc3-5386-4422-93cc-8dc7db09bb03
size(τ), size(pot_data["Q"][1,:])

# ╔═╡ d6dfc108-d25a-4a7c-838d-ac5e7da43c49
[-2π * dot(τ, pot_data["Q"][i,:]') for i=1:h11+4] .+ log(10)

# ╔═╡ 7ecff719-84c4-474e-a4cb-a3d38fe04e16
τ .== minimum(τ)

# ╔═╡ 51c5832f-afdd-4d76-8f4a-6b26fdc3e85e
md"""
The smallest PTD volume is $(minimum(τ))
"""

# ╔═╡ 1facdec9-789b-48c5-98e2-f522f8b75f0f
md"""
The largest instanton scale is 1e$(pot_data["L"][sortperm(pot_data["L"][:,2], rev=true),:][1,2])
"""

# ╔═╡ 8cd946f8-5909-4ed9-a5e7-1d11efded856
md"""
The smallest _leading_ instanton scale is 1e$(pot_data["L"][sortperm(pot_data["L"][1:h11+4,2], rev=true), :][end,2])
"""

# ╔═╡ 0c0b9219-f2e1-4333-b775-10a98d1fd798
begin
	threshold = 1e-2
	LQtilde = CYAxiverse.generate.LQtildebar(pot_data["L"],pot_data["Q"]; threshold=threshold);
	md"""
	In this system of $(size(pot_data["L"],1)) instantons generating a potential for $(size(pot_data["Q"],2)) axions, there are
	- $(size(LQtilde["Leff"],2)) leading instantons
	- and $(size(LQtilde["Qeff"],1)) axions with non-zero overlap satisfying the threshold of ``\frac{\Lambda_a}{\Lambda_j}`` ≥ $threshold
	"""
end

# ╔═╡ df395e06-f130-4dd2-a95e-799a1d6b7d1f
S = MatrixSpace(Nemo.ZZ, size(LQtilde["Qtilde"])...)

# ╔═╡ f939d71c-86ce-4e12-9e18-b112743a130e
m = S(LQtilde["Qtilde"])

# ╔═╡ 0df48817-9fdf-4032-8419-49fd940c89e8
Nemo.nullspace(m)

# ╔═╡ 77dd1f33-15b4-4885-9152-34427bcd7c4f
hcat(LQtilde["Ltilde"][:,1], LQtilde["Ltilde"][:,2])

# ╔═╡ 999ffbe3-a67e-4d6a-86ed-30687afb2da5
LQtilde["Ltilde"][:,sortperm(LQtilde["Ltilde"][2,:], rev = true)]

# ╔═╡ eb2b9729-af3c-412e-8133-e07fbcdaee49
hcat(LQtilde["Leff"][2,:],LQtilde["Qeff"][LQtilde["Qrowmask"],LQtilde["Qcolmask"]]')

# ╔═╡ 757b5ac0-5148-4980-ac09-d6e5e2930399
Matrix{Int}(LQtilde["Qtilde"]'[sortperm(LQtilde["Ltilde"][2,:], rev=true),:]')

# ╔═╡ b9733f0f-4869-4af5-9d65-2bb092e0bf54
size(hcat(LQtilde["Qtilde"][:,5], LQtilde["Qtilde"][:,11]))

# ╔═╡ 2fb703ac-e71a-4574-9aa6-eadef19ba8e6
LQtilde["Qtilde"][:,sortperm([sqrt(sum(abs2,i)) for i=eachcol(LQtilde["Qtilde"])], rev=true)]

# ╔═╡ 6307dc3a-8905-4773-b7fd-2daf60ed40f9
LQtilde["Qtilde"][:,1]

# ╔═╡ 676080a2-e02b-4632-89a1-e906d282051e
@. ifelse(mod(LQtilde["Qtilde"][:,1] / 2π, 1) ≈ 1 || mod(LQtilde["Qtilde"][:,1] / 2π, 1) ≈ 0, 0, LQtilde["Qtilde"][:,1])

# ╔═╡ 18eac169-749c-488d-8335-29f74e53975d
LQtilde["Qtilde"][:, [sum(i)!=5 for i in eachcol(LQtilde["Qtilde"])]]

# ╔═╡ 5b362ce3-614e-4b98-b4e7-cfdf5e3a1043
for i in eachcol(LQtilde["Qtilde"])
	i[:] = @. ifelse(mod(i / 2π, 1) ≈ 1 || mod(i / 2π, 1) ≈ 0, 0, i)
end

# ╔═╡ 363b43f0-33b8-466a-bcdf-fccfede32f98
LQtilde["Qbar"]'[1,:]

# ╔═╡ b0f33ae9-6056-4e0d-b79f-b7d880fd6016
md"""
The naïve estimate of vacua (_i.e._ taking the ``h^{1,1}`` leading, linearly-independent instantons as wholly dominant) is
``\mathrm{det}\tilde{\mathcal{Q}}`` = $(abs(det(LQtilde["Qtilde"][:,1:size(LQtilde["Qtilde"],1)])))
"""

# ╔═╡ 5035d43e-24fb-40b3-b928-0e544e6d62cd
algo_hz = Newton(alphaguess = LineSearches.InitialHagerZhang(α0=1.0), linesearch = LineSearches.HagerZhang());

# ╔═╡ bbc6437d-1010-49cc-ae82-0a247c8e0eec
algo_LBFGS = LBFGS(linesearch = LineSearches.BackTracking());

# ╔═╡ 5ede58ca-dec2-4419-837b-7307f28a2005
gradσ = CYAxiverse.minimizer.grad_std(h11,tri,cy);

# ╔═╡ 14945e1d-39c5-4975-ae0d-67d057f024d0
function minimize(LV::Vector,QV::Matrix,x0::Vector; algo, threshold = 1e-2)
    # setprecision(ArbFloat,digits=prec)
    # Arb0 = ArbFloat(0.)
    # Arb1 = ArbFloat(1.)
    # Arb2π = ArbFloat(2π)
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
        grad_temp = zeros(ArbFloat, size(LV,1),size(x,1))
        @tullio grad_temp[c,i] = QV[c,i] * sin.(QX(x)[c])
        @tullio gradient[i] = LV[c] * grad_temp[c,i]
    end
    function hess(x::Vector)
        grad2::Matrix{ArbFloat} = zeros(ArbFloat,(size(x,1),size(x,1)))
        hind1::Vector{Vector{Int64}} = [[x,y]::Vector{Int64} for x=1:size(x,1),y=1:size(x,1) if x>=y]
        grad2_temp::Vector{ArbFloat} = zeros(ArbFloat,size(hind1,1))
        grad2_temp1::Matrix{Float64} = zeros(Float64,size(LV,1),size(hind1,1))
        @tullio grad2_temp1[c,k] = @inbounds(begin
        i,j = hind1[k]
                QV[c,i] * QV[c,j] * cos.(QX(x)[c]) end) grad=false
        @tullio grad2_temp[k] = grad2_temp1[c,k] * LV[c]
        @inbounds for i in eachindex(hind1)
            j,k = hind1[i]
            grad2[j,k] = grad2_temp[i]
        end
        hessfull = Hermitian(grad2 + transpose(grad2) - Diagonal(grad2))
    end
    function hess!(hessian::Matrix, x::Vector)
        grad2 = zeros(ArbFloat,(size(x,1),size(x,1)))
        hind1 = [[x,y]::Vector{Int64} for x=1:size(x,1),y=1:size(x,1) if x>=y]
        grad2_temp = zeros(ArbFloat,size(hind1,1))
        grad2_temp1 = zeros(ArbFloat,size(LV,1),size(hind1,1))
        @tullio grad2_temp1[c,k] = @inbounds(begin
                i,j = hind1[k]
                QV[c,i] * QV[c,j] * cos.(QX(x)[c]) end) grad=false avx=false
        @tullio grad2_temp[k] = grad2_temp1[c,k] * LV[c]
        @inbounds for i in eachindex(hind1)
            j,k = hind1[i]
            grad2[j,k] = grad2_temp[i]
        end
        hessian .= grad2 + transpose(grad2) - Diagonal(grad2)
    end
    grad(x) = vcat([dot(LV,QV[:,i] .* sin.(QX(x))) for i ∈ 1:size(x,1)]...)
    res = optimize(fitness,grad!,hess!,
                x0, algo,
                Optim.Options(x_tol =minimum(abs.(LV)),g_tol =minimum(threshold .* abs.(gradσ))))
    Vmin = Optim.minimum(res)
    xmin = Optim.minimizer(res)
    GC.gc()
    # if Float64(log10(abs(minimum(eigen(hess(xmin)).values)))) < -prec && sum(Float64.(log10.(abs.(grad(xmin)))) .< log10.(abs.(threshold .* gradσ))) == (h11 - size(gradσ[gradσ .== 0.],1))
    hess_eigs = Float64(log10(abs(minimum(eigen(hess(xmin)).values)))) 
    hess_sign = sign((minimum(eigen(hess(xmin)).values)))
    sum_grad = sum(Float64.(log10.(abs.(grad(xmin)))))
    Vmin_sign = Int(sign(Vmin))
    Vmin_log = Float64(log10(abs(Vmin)))
    xmin_log = Float64.(log10.(abs.(xmin)))
    xmin_sign = Int.(sign.(xmin))

    keys = ["±V", "logV","±x", "logx", "Heigs", "Hsign", "gradsum"]
    vals = [Vmin_sign, Vmin_log, xmin_sign, xmin_log, hess_eigs, hess_sign, sum_grad]
    return Dict(zip(keys,vals))
    GC.gc()
    # end
end

# ╔═╡ 316f3ae8-0110-4514-b132-38ddaa56b1ef
md"""
The smallest gradient element after a scan of 100 random points in the potential is
≈ 1e$(Float64(log10(minimum(gradσ[gradσ .> 0]))))
"""

# ╔═╡ 42445c21-0608-4330-b04d-0ee49f592318
Lfull = ArbFloat.(LQtilde["Leff"][1,:]) .* ArbFloat(10.) .^ ArbFloat.(LQtilde["Leff"][2,:]);

# ╔═╡ 4fb9299e-564a-467c-8b15-9ebd1438b135
md"""
The size of ``\mathcal{Q}_\mathrm{eff}`` is $(size(LQtilde["Qeff"]')) reduced from $(size(pot_data["Q"]))
"""

# ╔═╡ 2eb2a848-14f3-4b10-a93f-d845b8d4daab
Markdown.parse("""
The axions that contribute to the _effective_ or _reduced_ potential are ``\\tilde{\\theta}_i`` for i ∈ $((1:size(LQtilde["Qtilde"],1))[LQtilde["Qrowmask"]])
""")

# ╔═╡ 5bae78d1-8cf6-415c-8608-cc187a80759c
begin
	Markdown.parse("""
	The instantons that generate the _effective_ or _reduced_ potential are ``\\tilde{\\Lambda}_a`` for a ∈ $(collect(1:size(LQtilde["Qtilde"],2))[LQtilde["Qcolmask"]])
	""")
end

# ╔═╡ 89873528-0803-4a26-8cae-f560e8e3090f
begin
	min_data = []
	@progress for i=1:500
		x0 = ArbFloat.(rand(Uniform(0,2π),size(LQtilde["Qeff"],1))) .* rand(ArbFloat,size(LQtilde["Qeff"],1))
		global prec = Int(round(abs(minimum(LQtilde["Leff"][2,:]))))+10
		min_data_temp = CYAxiverse.minimizer.minimize(h11,tri,cy,Lfull ./ maximum(Lfull),Matrix{ArbFloat}(LQtilde["Qeff"]'),x0,gradσ,algo_hz, prec < 1_000 ? prec : 1_000)
		if min_data_temp["Hsign"] > 0
			push!(min_data,min_data_temp)
		end
		if size(min_data,1) == 200
			break
		end
		global num_limit = i
	end
	md"""
	**There are $(size(min_data,1)) minima found from $(num_limit) numerical searches to $(min(1_000,prec)) significant figures**
	"""
end

# ╔═╡ a7d02d1c-01de-4b37-908d-0683cd7117a0
[LQtilde["Leff"][2,:][Vector{Bool}(abs.(LQtilde["Qeff"][i,:]))] for i=1:size(LQtilde["Qeff"],1)]

# ╔═╡ e96e7f3f-802f-421c-a4e4-f60aa18f646d
size(LQtilde["Qtilde"][:,1][LQtilde["Qtilde"][:,1] .== 0])

# ╔═╡ 6030dcb9-16c9-42f9-a214-286784de3100
LQtilde["Qtilde"][:,9]

# ╔═╡ 8d5d3d0e-b156-4e0b-b3c7-7e558ee0230e
[i for (i,item) = enumerate(eachcol(LQtilde["Qtilde"])) if size(item[item .== 0],1) != h11-1]

# ╔═╡ c703a028-4686-4f6e-a3b5-75cc71f1b434
size(LQtilde["Qrowmask"]), size(LQtilde["Qcolmask"])

# ╔═╡ 41e2169d-65ff-44cb-9828-68e1c4a530d2
function potential(L,Q,x)
	V = dot(L,(ArbFloat(1.) .- cos.(Q' * x)))
end;

# ╔═╡ 70b9770a-57c3-4ad4-90b8-f4efabc04ed0
function grad_potential(L, Q, x::Number)
	∂V = dot(L, Q' .* sin.(Q' * x))
end

# ╔═╡ 9775a872-8a9a-4286-8ae7-c34c6560ed39
function grad_potential(L, Q, x::Vector)
	Q1 = Q[1, :]'
	Q2 = Q[2, :]'
	θ1 = x[1]
	θ2 = x[2]
	∂V1 = dot(L, Q1 .* sin.(Q1 * θ1))
	∂V2 = dot(L, Q2 .* sin.(Q2 * θ2))
	[∂V1, ∂V2]
end

# ╔═╡ e1e6d095-f11a-40b7-8dc4-51952c4bdb30
LQtilde["Qrowmask"]

# ╔═╡ 7801297e-c52e-48e0-92d6-d58839afe7e3
begin
	xtest = [ArbFloat.(min_data[i]["±x"] .* 10. .^ min_data[i]["logx"]) for i in eachindex(min_data)];
	xmin_test = [[LQtilde["Ltilde"][1,i] < 0. ? ArbFloat(π) : ArbFloat(0.) for i=1:size(LQtilde["Qtilde"],1)] for _ in xtest]
	for i in eachindex(xtest)
		for (j,k) in enumerate((1:size(LQtilde["Qtilde"],1))[LQtilde["Qrowmask"]])
			xmin_test[i][k] = xtest[i][j]
		end
		xtest[i] = inv(LQtilde["Qtilde"][:,1:size(LQtilde["Qtilde"],1)]') * xmin_test[i] ##Should this be Qtilde' or not in my code?!
	end
end

# ╔═╡ fd81eff2-d87b-4a5c-945d-8b9096ee5737
xmin_test[1]

# ╔═╡ 2fbee316-ca49-46c5-9afe-c1fe561ea948
Ltest = ArbFloat.(LQtilde["Ltilde"][1,:] .* 10. .^ LQtilde["Ltilde"][2,:]);

# ╔═╡ a9189c10-9aa6-4a07-abf5-7cfc9ec3ab62
Qtest = ArbFloat.(Matrix{Float64}(LQtilde["Qtilde"]));

# ╔═╡ 910dc6ba-9ad0-4f19-addb-81d882d6322c
xs = range(-0.5π,2.5π,length=300);

# ╔═╡ a45e6c9d-b616-48c9-a7c2-0afe9d835516
ys = xs;

# ╔═╡ ebff481b-fcc3-4644-b766-c810c5131892
md"""
Show 1D or 2D plot?
"""

# ╔═╡ 27e7776b-1443-4cf0-b5d4-3985d89b85b5
@bind plot_dims confirm(PlutoUI.Select(["1D","2D"]))

# ╔═╡ 2a9e079d-4131-4e2e-a6ce-e3111cbcb7d3
LQtilde["Qeff"]

# ╔═╡ 3a918852-9ec4-48f9-a2c5-b2de8a3b7e7a
LQtilde["Ltilde"]

# ╔═╡ 4e6bbc87-bc60-47c2-960e-1084afed680f
hcat(xtest...)

# ╔═╡ ee2060d2-00da-4ede-998e-23bd7805efc9
[i for i=1:h11 if mod.((Float64.((hcat(xtest...)[i,:])) ./ 2π), 1) != zeros(size(xtest,1)) .|| any(mod.((Float64.((hcat(xtest...)[i,:])) ./ 2π), 1) == 1.)]

# ╔═╡ 732bc5bb-419b-4361-a331-b563d15f1db1
LQtilde["Qtilde"][:,[i for i=1:h11 if mod.((Float64.((hcat(xtest...)[i,:])) ./ 2π), 1) != zeros(size(xtest,1)) .|| any(mod.((Float64.((hcat(xtest...)[i,:])) ./ 2π), 1) == 1.)]] * LQtilde["Ltilde"][2,[i for i=1:h11 if mod.((Float64.((hcat(xtest...)[i,:])) ./ 2π), 1) != zeros(size(xtest,1)) .|| any(mod.((Float64.((hcat(xtest...)[i,:])) ./ 2π), 1) == 1.)]]

# ╔═╡ c8507013-7fe1-4558-8992-53ecb650fe4d
md"""
Set which axion direction(s) to plot:
"""

# ╔═╡ 4d96cbc5-f27c-4cf8-aac9-f73988761a09
if plot_dims == "1D"
	@bind axion confirm(PlutoUI.Select((1:size(LQtilde["Qtilde"],1))))
		# [hcat(xtest...)[:,1] .!= 0. .&& abs.(hcat(xtest...)[:,1]) .!= Float64(π),:][:,1]))
else
	@bind axion confirm(PlutoUI.Select([[i,j] for i in (1:size(LQtilde["Qtilde"],1)), j in (1:size(LQtilde["Qtilde"],1)) if i<j]))
end

# ╔═╡ b54d02e9-634d-4451-a942-fc0a57c935d3
if plot_dims == "1D"
	V = [potential(Ltest,Qtest[axion,:],θ1) for θ1 in xs];
else
	V = [potential(Ltest,hcat(Qtest[axion[1],:],Qtest[axion[2],:])',[θ1,θ2]) for θ1 in xs, θ2 in ys];
end;

# ╔═╡ 5437ccfc-224f-4744-b08a-994460f7f2bb
if plot_dims == "1D"
	∂V = [grad_potential(Ltest,Qtest[axion,:],θ1) for θ1 in xs];
else
	# ∂V = [grad_potential(Ltest,hcat(Qtest[axion[1],:],Qtest[axion[2],:])',[θ1,θ2]) for θ1 in xs, θ2 in ys];
	∂V1 = [grad_potential(Ltest,Qtest[axion[1],:],θ1) for θ1 in xs];
	∂V2 = [grad_potential(Ltest,Qtest[axion[2],:],θ2) for θ2 in ys];
end;

# ╔═╡ a422be5b-4031-4f34-b0db-a1c380d61351
if plot_dims == "1D"
	Vmin = [potential(Ltest,Qtest[axion,:],θ1) for θ1 in hcat(xtest...)[axion,:]];
else
	Vmin = [potential(Ltest,hcat(Qtest[axion[1],:],Qtest[axion[2],:])',[θ1,θ2]) for θ1 in hcat(xtest...)[axion[1],:], θ2 in hcat(xtest...)[axion[2],:]];
end;

# ╔═╡ 1c32e0b9-1243-44d1-9a4f-54d966abbee9
if plot_dims == "1D"
	∂Vmin = [grad_potential(Ltest,Qtest[axion,:],θ1) for θ1 in hcat(xtest...)[axion,:]];
else
	# ∂Vmin = [grad_potential(Ltest,hcat(Qtest[axion[1],:],Qtest[axion[2],:])',[θ1,θ2]) for θ1 in hcat(xtest...)[axion[1],:], θ2 in hcat(xtest...)[axion[2],:]];
	∂Vmin1 = [grad_potential(Ltest,Qtest[axion[1],:],θ1) for θ1 in hcat(xtest...)[axion[1],:]];
	∂Vmin2 = [grad_potential(Ltest,Qtest[axion[2],:],θ2) for θ2 in hcat(xtest...)[axion[2],:]];
end;

# ╔═╡ 16c15a91-7f4d-4d13-8a9f-2c155cf31037
precision(ArbFloat)

# ╔═╡ c1b94e33-c604-42b9-a103-a551fb1ca93a
begin
	fig = Figure(resolution = (1200,800))
	if V isa(Matrix)
		ax1 = Axis(fig[1:2, 1], xlabel = L"\frac{\theta_{%$(axion[1])}}{2\pi}", ylabel = L"\frac{\theta_{%$(axion[2])}}{2\pi}")
		hmV = CairoMakie.heatmap!(ax1, xs ./ 2π, ys ./ 2π, log10.(abs.(V)))
		CairoMakie.Colorbar(fig[:, end+1],hmV, label = L"\log_{10}\,V(\vec{\theta})")
		CairoMakie.scatter!(ax1,mod.((Float64.((hcat(xtest...)[axion[1],:])) ./ 2π),1),mod.((Float64.((hcat(xtest...)[axion[2],:])) ./ 2π),1), color = :red, markersize = 5)

		ax2 = Axis(fig[1, end+1], xlabel = L"\frac{\theta_{%$(axion[1])}}{2\pi}", ylabel = L"\log_{10}\,\del V(\theta_{%$(axion[1])})")
		lines!(ax2, xs ./ 2π, ∂V1 == zeros(size(∂V1,1)) ? minimum(∂V1) .* ones(size(∂V1,1)) : log10.(abs.(∂V1)), linewidth = 3, color = log10.(abs.(V[:,1])))
		CairoMakie.scatter!(ax2, mod.((Float64.((hcat(xtest...)[axion[1],:])) ./ 2π), 1),∂Vmin1 == zeros(size(∂Vmin1,1)) ? minimum(∂Vmin1) .* ones(size(∂Vmin1,1)) : log10.(abs.(∂Vmin1)), color = :red, markersize = 5)

		ax3 = Axis(fig[2, end], xlabel = L"\frac{\theta_{%$(axion[2])}}{2\pi}", ylabel = L"\log_{10}\,\del V(\theta_{%$(axion[2])})")
		lines!(ax3, ys ./ 2π, ∂V2 == zeros(size(∂V2,1)) ? minimum(∂V2) .* ones(size(∂V2,1)) : log10.(abs.(∂V2)), linewidth = 3, color = log10.(abs.(V[1,:])))
		CairoMakie.scatter!(ax3, mod.((Float64.((hcat(xtest...)[axion[2],:])) ./ 2π), 1),∂Vmin2 == zeros(size(∂Vmin2, 1)) ? minimum(∂Vmin2) .* ones(size(∂Vmin2,1)) : log10.(abs.(∂Vmin2)), color = :red, markersize = 5)
		# xlims!(-0.1,1.1)
		# ylims!(-0.1,1.1)
		fig
	elseif V isa(Vector)
		ax1 = Axis(fig[1,1], xlabel = L"\frac{\theta_{%$(axion[1])}}{2\pi}", ylabel = L"V(\theta_{%$(axion[1])})")
		lines!(ax1, xs ./ 2π,V == zeros(size(V,1)) ? minimum(V) .* ones(size(V,1)) : log10.(abs.(V)), linewidth = 3, color = log10.(abs.(V)))
		CairoMakie.scatter!(ax1, mod.((Float64.((hcat(xtest...)[axion,:])) ./ 2π), 1),Vmin == zeros(size(Vmin,1)) ? minimum(Vmin) .* ones(size(Vmin,1)) : log10.(abs.(Vmin)), color = :red, markersize = 5)
		
		ax2 = Axis(fig[1,2], xlabel = L"\frac{\theta_{%$(axion[1])}}{2\pi}", ylabel = L"\log_{10}\,\del V(\theta_{%$(axion)})")
		lines!(ax2, xs ./ 2π,∂V == zeros(size(∂V,1)) ? minimum(∂V) .* ones(size(∂V,1)) : log10.(abs.(∂V)), linewidth = 3, color = log10.(abs.(V)))
		CairoMakie.scatter!(ax2, mod.((Float64.((hcat(xtest...)[axion,:])) ./ 2π), 1),∂Vmin == zeros(size(∂Vmin,1)) ? minimum(∂Vmin) .* ones(size(∂Vmin,1)) : log10.(abs.(∂Vmin)), color = :red, markersize = 5)
		# xlims!(-0.5π,2.5π)
		# ylims!(-0.5π,2.5π)
		fig
	end
end
	

# ╔═╡ 2b382a1a-fd60-4ecd-b550-c396d7496d0f


# ╔═╡ 8e30171e-0146-450e-b31d-26a18adb2fe3
md"""
The upper limit to the number of vacua in this system is $(floor(sqrt(det(LQtilde["Qtilde"] * LQtilde["Qtilde"]'))))
"""

# ╔═╡ 3b5c385d-fd90-4b49-8155-0d28582a1384
# ╠═╡ disabled = true
#=╠═╡
@time CYAxiverse.generate.vacua_TB(pot_data["L"],pot_data["Q"]; threshold=1e-2)["vacua"]
  ╠═╡ =#

# ╔═╡ f11dfbf2-7d3a-4d90-b6f9-3fbf8cc6f764
# ╠═╡ disabled = true
#=╠═╡
CYAxiverse.generate.LQtildebar(8,108,1; threshold=0.01)
  ╠═╡ =#

# ╔═╡ bcc4d307-8aa5-497b-af76-ad000fd62b72
# ╠═╡ disabled = true
#=╠═╡
h11list[:,16_001], h11list[:,47_000]
  ╠═╡ =#

# ╔═╡ d2ec3683-4e96-41cd-8e6b-51f45c31df89
# ╠═╡ disabled = true
#=╠═╡
begin 
	onemin = [];
	notonemin = [];
	for i=rand(1:size(h11list,2),1000)
		h11,tri,cy = h11list[:,i]
		LQtildebar_data = CYAxiverse.generate.LQtildebar(h11,tri,cy; threshold=0.01)
		Qtilde = LQtildebar_data["Qtilde"]
		if size(Qtilde,1) != size(Qtilde,2)
			push!(notonemin,[h11,tri,cy,size(LQtildebar_data["Qeff"])...])
		else
			push!(onemin,[h11,tri,cy])
		end
	end
end
  ╠═╡ =#

# ╔═╡ 61dae36d-46e2-45db-b6a1-90132febf8a6
# ╠═╡ disabled = true
#=╠═╡
onemin_list = hcat(onemin...)
  ╠═╡ =#

# ╔═╡ 2fe9315e-90d5-4e56-844b-61fde4902091
# ╠═╡ disabled = true
#=╠═╡
notonemin_list = hcat(notonemin...)
  ╠═╡ =#

# ╔═╡ 054dfb58-c816-48d7-bb12-9d43e1715efb
# ╠═╡ disabled = true
#=╠═╡
notonemin_list[:,notonemin_list[4,:] .> notonemin_list[1,:] .- 10]
  ╠═╡ =#

# ╔═╡ 766cffea-b562-4997-9b2a-f05a27c59c2e
# ╠═╡ disabled = true
#=╠═╡
@time test_mass = CYAxiverse.generate.pq_spectrum(pot_data["K"],pot_data["L"],pot_data["Q"])
  ╠═╡ =#

# ╔═╡ 9d2af162-ce30-475b-921b-766e34692fc4
# ╠═╡ disabled = true
#=╠═╡
@time test_spec = CYAxiverse.generate.hp_spectrum(pot_data["K"],pot_data["L"],pot_data["Q"]; prec=5_000)
  ╠═╡ =#

# ╔═╡ 962576a8-fef2-48c4-b7fa-48ff5424298c
# ╠═╡ disabled = true
#=╠═╡
test_spec["m"][(test_spec["m"] .- test_mass["m"]) .== maximum(test_spec["m"] .- test_mass["m"])]
  ╠═╡ =#

# ╔═╡ c7eeaa51-71b9-4b40-85cc-02063c64b8ce
# ╠═╡ disabled = true
#=╠═╡
test_mass["m"][(test_spec["m"] .- test_mass["m"]) .== maximum(test_spec["m"] .- test_mass["m"])]
  ╠═╡ =#

# ╔═╡ 5f8113f3-59c3-42d4-8544-a61838381407
# ╠═╡ disabled = true
#=╠═╡
for t=4:100
	n = lpad(t,3,"0")
	h5open("/scratch/users/mehta2/vacua_0822/h11_$n/np_0000001/cy_0000001/cyax.h5","r") do file
		if haskey(file, "cytools/geometric/h21")
			println("Gone too far")
		else
		[[string("$i/",keys(file[i]),"/",keys(file[i][j])) for j in keys(file[i])] for i in keys(file)]
		end
	end
end
  ╠═╡ =#

# ╔═╡ 4dea5c0f-fb4e-4510-9aab-902658407641
# ╠═╡ disabled = true
#=╠═╡
CYAxiverse.generate.LQtildebar(pot_data["L"],pot_data["Q"]; threshold=0.5)
  ╠═╡ =#

# ╔═╡ Cell order:
# ╟─fee399f9-2668-41e0-a296-37b348a04769
# ╟─90f44877-6310-49b1-9331-f8601918e4b3
# ╟─2000a078-38f5-4c93-8627-ba6b4970aef6
# ╠═7c8e7502-94d8-4da6-a5e2-b950b33a62c2
# ╠═7fb60b52-1158-4bcb-b5e0-1e3ebd1ad52b
# ╠═d1f78454-3ba0-4569-a491-f52e737c7dc3
# ╠═3788df6d-c756-4b6a-8d75-8cd018ab2991
# ╠═9071d014-a286-4f6f-bafc-648c1954d3d7
# ╠═25e1eb34-0e4b-4bf2-ac1b-228545bb82a8
# ╟─4c1b071d-65ad-41ce-b7ab-9b11fcf15ce5
# ╠═e1fdc533-efc0-4012-8464-3db601f66819
# ╠═8699f77c-4a59-4194-9f08-f878b65b93ba
# ╠═281b0be2-86fd-4201-871a-7e362b5872d7
# ╟─871778ce-735d-49ed-b6f5-ab68f127ad32
# ╟─0b13b99a-9c7d-4117-90ba-e74baf49054d
# ╠═df395e06-f130-4dd2-a95e-799a1d6b7d1f
# ╠═f939d71c-86ce-4e12-9e18-b112743a130e
# ╠═0df48817-9fdf-4032-8419-49fd940c89e8
# ╠═4cfe15ea-1a7f-4af3-9602-6eac3732232a
# ╠═0b261480-4109-41f9-ac4e-52d040b20efe
# ╠═33fc2f88-c402-4394-83c9-10df2e7a6d96
# ╠═6c6bd3c6-4019-426c-9188-e63debe0fe8b
# ╠═b5020c4c-d37f-4ad0-9633-ebdef6898afc
# ╠═1c45d205-f8fb-4d1f-8d57-0d6d46ce622f
# ╠═fd53a8a1-c782-4f31-bf8a-1403de56b64d
# ╠═4993e5d8-48f1-4b4e-878c-9576e7bfc85c
# ╠═5abd2caa-8d2d-4c8b-b064-a0426447b392
# ╠═07f43035-2e72-47b1-baef-7258c65a13c8
# ╠═64f3ac82-0f2b-4092-b904-b750ec16a309
# ╠═6f614ec4-c03f-4d65-bd3b-153a31c23da4
# ╠═622570d7-a3fa-4bd0-8d1d-5762490e8ad0
# ╠═6c5a1870-3657-4978-995f-578fe9d0602c
# ╠═cd6bbbd5-321f-4868-b0e7-71b96965bef7
# ╠═77dd1f33-15b4-4885-9152-34427bcd7c4f
# ╠═999ffbe3-a67e-4d6a-86ed-30687afb2da5
# ╠═eb2b9729-af3c-412e-8133-e07fbcdaee49
# ╠═757b5ac0-5148-4980-ac09-d6e5e2930399
# ╠═6f9c2ba1-6536-44d2-938e-7adf68b93bc5
# ╠═b9733f0f-4869-4af5-9d65-2bb092e0bf54
# ╠═14945e1d-39c5-4975-ae0d-67d057f024d0
# ╠═2fb703ac-e71a-4574-9aa6-eadef19ba8e6
# ╠═a1dfa08c-1056-4e15-a040-4de1276f400c
# ╠═6307dc3a-8905-4773-b7fd-2daf60ed40f9
# ╠═676080a2-e02b-4632-89a1-e906d282051e
# ╠═18eac169-749c-488d-8335-29f74e53975d
# ╠═5b362ce3-614e-4b98-b4e7-cfdf5e3a1043
# ╠═ecfc017c-5eca-41b2-ba3e-71d6dbe68403
# ╠═363b43f0-33b8-466a-bcdf-fccfede32f98
# ╠═dc4c35f6-90cc-4a71-ab8d-b9ae847ca561
# ╠═ac863bc3-5386-4422-93cc-8dc7db09bb03
# ╠═d6dfc108-d25a-4a7c-838d-ac5e7da43c49
# ╠═7ecff719-84c4-474e-a4cb-a3d38fe04e16
# ╠═51c5832f-afdd-4d76-8f4a-6b26fdc3e85e
# ╟─1facdec9-789b-48c5-98e2-f522f8b75f0f
# ╟─8cd946f8-5909-4ed9-a5e7-1d11efded856
# ╟─0c0b9219-f2e1-4333-b775-10a98d1fd798
# ╟─b0f33ae9-6056-4e0d-b79f-b7d880fd6016
# ╟─5035d43e-24fb-40b3-b928-0e544e6d62cd
# ╟─bbc6437d-1010-49cc-ae82-0a247c8e0eec
# ╠═5ede58ca-dec2-4419-837b-7307f28a2005
# ╟─316f3ae8-0110-4514-b132-38ddaa56b1ef
# ╠═42445c21-0608-4330-b04d-0ee49f592318
# ╟─4fb9299e-564a-467c-8b15-9ebd1438b135
# ╟─2eb2a848-14f3-4b10-a93f-d845b8d4daab
# ╟─5bae78d1-8cf6-415c-8608-cc187a80759c
# ╠═89873528-0803-4a26-8cae-f560e8e3090f
# ╠═a7d02d1c-01de-4b37-908d-0683cd7117a0
# ╠═e96e7f3f-802f-421c-a4e4-f60aa18f646d
# ╠═6030dcb9-16c9-42f9-a214-286784de3100
# ╠═8d5d3d0e-b156-4e0b-b3c7-7e558ee0230e
# ╠═c703a028-4686-4f6e-a3b5-75cc71f1b434
# ╠═41e2169d-65ff-44cb-9828-68e1c4a530d2
# ╠═70b9770a-57c3-4ad4-90b8-f4efabc04ed0
# ╠═9775a872-8a9a-4286-8ae7-c34c6560ed39
# ╠═e1e6d095-f11a-40b7-8dc4-51952c4bdb30
# ╠═7801297e-c52e-48e0-92d6-d58839afe7e3
# ╠═fd81eff2-d87b-4a5c-945d-8b9096ee5737
# ╠═2fbee316-ca49-46c5-9afe-c1fe561ea948
# ╠═a9189c10-9aa6-4a07-abf5-7cfc9ec3ab62
# ╠═910dc6ba-9ad0-4f19-addb-81d882d6322c
# ╠═a45e6c9d-b616-48c9-a7c2-0afe9d835516
# ╟─ebff481b-fcc3-4644-b766-c810c5131892
# ╠═27e7776b-1443-4cf0-b5d4-3985d89b85b5
# ╠═2a9e079d-4131-4e2e-a6ce-e3111cbcb7d3
# ╠═3a918852-9ec4-48f9-a2c5-b2de8a3b7e7a
# ╠═4e6bbc87-bc60-47c2-960e-1084afed680f
# ╠═ee2060d2-00da-4ede-998e-23bd7805efc9
# ╠═732bc5bb-419b-4361-a331-b563d15f1db1
# ╠═c8507013-7fe1-4558-8992-53ecb650fe4d
# ╠═4d96cbc5-f27c-4cf8-aac9-f73988761a09
# ╠═b54d02e9-634d-4451-a942-fc0a57c935d3
# ╠═5437ccfc-224f-4744-b08a-994460f7f2bb
# ╠═a422be5b-4031-4f34-b0db-a1c380d61351
# ╠═1c32e0b9-1243-44d1-9a4f-54d966abbee9
# ╠═16c15a91-7f4d-4d13-8a9f-2c155cf31037
# ╠═c1b94e33-c604-42b9-a103-a551fb1ca93a
# ╠═2b382a1a-fd60-4ecd-b550-c396d7496d0f
# ╟─8e30171e-0146-450e-b31d-26a18adb2fe3
# ╠═3b5c385d-fd90-4b49-8155-0d28582a1384
# ╠═f11dfbf2-7d3a-4d90-b6f9-3fbf8cc6f764
# ╠═bcc4d307-8aa5-497b-af76-ad000fd62b72
# ╠═d2ec3683-4e96-41cd-8e6b-51f45c31df89
# ╠═61dae36d-46e2-45db-b6a1-90132febf8a6
# ╠═2fe9315e-90d5-4e56-844b-61fde4902091
# ╠═054dfb58-c816-48d7-bb12-9d43e1715efb
# ╠═766cffea-b562-4997-9b2a-f05a27c59c2e
# ╠═9d2af162-ce30-475b-921b-766e34692fc4
# ╠═962576a8-fef2-48c4-b7fa-48ff5424298c
# ╠═c7eeaa51-71b9-4b40-85cc-02063c64b8ce
# ╠═5f8113f3-59c3-42d4-8544-a61838381407
# ╠═4dea5c0f-fb4e-4510-9aab-902658407641
