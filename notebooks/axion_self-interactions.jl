### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# ╔═╡ 3788df6d-c756-4b6a-8d75-8cd018ab2991
using Pkg; 

# ╔═╡ fbb69bcb-64c6-42c2-8ce1-666f397eb40e
# ╠═╡ show_logs = false
Pkg.activate("/scratch/users/mehta2/cyaxiverse/CYAxiverse");

# ╔═╡ b1362f7d-55e5-48d6-a695-f3bf59d8bf99
using PlutoUI, HDF5, ArbNumerics, LineSearches, Optim, CairoMakie, Distributions, LinearAlgebra, ProgressLogging, Revise, Random, SparseArrays, NLsolve, NormalForms, DifferentialEquations, IntervalArithmetic, IntervalRootFinding, StaticArrays, Nemo, ColorSchemes, Dates

# ╔═╡ e556408b-25f7-4fae-ba0b-243242279ba8
# ╠═╡ show_logs = false
@time using CYAxiverse

# ╔═╡ 8778a5d2-5eae-426b-bc86-c62c9326c9fd
begin
	using PyCall
	PyCall.current_python()
end

# ╔═╡ fee399f9-2668-41e0-a296-37b348a04769
md"""
# Vacua search -- Numerics
!!! update
	### September 2023
"""

# ╔═╡ 90f44877-6310-49b1-9331-f8601918e4b3
md"""
### This notebook is for rewriting the ``\lambda_{ijkl}`` computation.
"""

# ╔═╡ 915e345e-7002-489c-8fec-8395381f0fe5
md"""
!!! note
	It seems that if Log10(|Λ̄|) - Log10(|Λ̃|) ≳ 0.6, local vacua *do not* appear
"""

# ╔═╡ 2000a078-38f5-4c93-8627-ba6b4970aef6
html"""
<style>
	main {
    max-width: 70%;
    align-self: flex-start;
    margin-left: 70px;
}

pluto-helpbox {
	width: clamp(300px,calc(100vw - 781px),600px)
}
</style>
"""

# ╔═╡ 7c8e7502-94d8-4da6-a5e2-b950b33a62c2
begin
	ENV["newARGS"] = string("vacua_0323")
	md"""
	### The database we are using is: $(ENV["newARGS"])
	"""
end


# ╔═╡ 8c7bb44d-edb3-46b3-aeef-ac21d2ee16f5
begin
	h11list = CYAxiverse.filestructure.paths_cy()[2]
	md"""
	#### There are a total of $(size(h11list,2)) geometries in the database with $(minimum(Set(h11list[1,:]))) ≤ ``h^{1,1}`` ≤ $(maximum(Set(h11list[1,:])))
	"""
end

# ╔═╡ bfd0a313-a144-4883-abee-e74ac7f4a8e4
@time begin
	h11 = 15
	Ltest = [10. ^-i for i in 0:h11]
	Lsign = ones(h11+1)
	Qtest = Int.(I(h11))
	Qtest = hcat(Qtest, zeros(h11))
	Qtest[1, end] = 1
	size(Qtest), size(Ltest)
end

# ╔═╡ 1e3c5202-df60-47c5-84df-31fee243fe80
Ltest

# ╔═╡ b07db615-e3c0-4d60-ab3f-13f35b99f1c9
@time begin
	Ktest = rand(h11, h11)
	Ktest = Hermitian(1/2 .* Ktest'*Ktest + 2I(h11))
end

# ╔═╡ eb35a641-b828-42fe-b01e-ecd6ccc7212f
CYAxiverse.generate.pq_spectrum(Ktest, hcat(Lsign, Ltest), Matrix{Int}(Qtest')), CYAxiverse.generate.hp_spectrum(Ktest, hcat(Lsign, Ltest), Matrix{Int}(Qtest'))

# ╔═╡ 2b3a2af1-091b-47df-bc2f-082b00a4342b
function pq_spectrum_square(α::CYAxiverse.structs.CanonicalQBasis)
    Ltilde = α.Lhat
    Qtilde = α.Qhat
    QKs::Matrix{Float64} = zeros(Float64,h11,h11)
    fapprox::Vector{Float64} = zeros(Float64,h11)
    mapprox::Vector{Float64} = zeros(h11)
    LinearAlgebra.mul!(QKs, inv(Kls'), Matrix(Qtilde'))
    for i=1:h11
        println(size(QKs[i, :]))
        fapprox[i] = log10(1/(2π*dot(QKs[i,:],QKs[i,:])))
        mapprox[i] = 0.5*(Ltilde[2,i]-fapprox[i])
        T = orth_basis(QKs[i,:])
        QKs1 = zeros(size(QKs,1), size(T,2))
        LinearAlgebra.mul!(QKs1,QKs, T)
        println(size(QKs1))
        # Qlt[i, :] .= QKs[i, :]
        QKs = deepcopy(QKs1)
    end
	AxionSpectrum(mapprox[sortperm(mapprox)] .+ 9. .+ Float64(log10(constants()["MPlanck"])), 0.5 .* fapprox[sortperm(mapprox)] .+ Float64(log10(constants()["MPlanck"])), fK .+ Float64(log10(constants()["MPlanck"])) .- Float64(constants()["log2π"]))
end

# ╔═╡ 4476d3fe-db5b-4ed5-aaad-5915b2eb1605
function pq_spectrum(K::Hermitian{Float64, Matrix{Float64}}, L::Matrix{Float64}, Q::Matrix{Int}; threshold = 0.5)
    h11::Int = size(K,1)
    fK::Vector{Float64} = log10.(sqrt.(eigen(K).values))
    Kls = cholesky(K).L
    LQ = CYAxiverse.generate.LQtilde(Q', L')
	α = CYAxiverse.generate.:αmatrix(LQ; threshold = threshold)
	if typeof(α)<:Canonicalα
	else
	    pq_spectrum_square(α)
	end
end

# ╔═╡ 8ff2f070-1ae8-4c0f-b540-34e76fe0d685
αtest = CYAxiverse.generate.:αmatrix(CYAxiverse.structs.GeometryIndex(10, 40, 1))

# ╔═╡ 4e965d07-15dd-4840-b034-2a997a935567
qr(Qtest[1,:]).Q, CYAxiverse.generate.orth_basis(Vector(Qtest[1, :]))

# ╔═╡ d98e8576-8bf0-4a59-ad29-80b4ddba366e
@time begin
	Random.seed!(1234567890)
	Qtemp = rand(1:10, 10,10)
	# Qtemp = Qtemp + transpose(Qtemp) - Diagonal(Qtemp)
	eigen(CYAxiverse.generate.hessian(zeros(10), hcat(ones(10), [-i for i in 2:11]), Qtemp)).values
end

# ╔═╡ f915ce5a-56b9-4634-967c-b955c2246419
eigen(Ltest[2:11] .* CYAxiverse.generate.hessian_norm(zeros(10), Qtemp)).values

# ╔═╡ 8dd2c1c6-620d-431f-9ece-be204bbab7eb
begin
	Random.seed!(1234567890)
	Qtemp1 = Matrix(hcat(I(h11), rand(-5:5, h11)))
	Qtemp1
end

# ╔═╡ f35bfd5e-015f-49e8-a87d-e00320785fee
@time begin
	canK = cholesky(Ktest).L
	test_hess = zeros(size(Qtemp1, 1), size(Qtemp1, 1))
	for (i,q) in enumerate(eachcol(inv(canK)' * Qtemp1))
		test_hess .+= Ltest[i] * (q*q') 
		# * cos.(zeros(size(Qtemp1,1))' * (inv(canK)' * Qtemp1))[i]
	end
	# test_hess = Hermitian(test_hess + test_hess' - Diagonal(test_hess))
end

# ╔═╡ a6c24db6-367a-4a2a-9eed-47c00101a099
inv(canK)' * Qtemp1

# ╔═╡ 5daa448d-d9e7-4d43-b68e-b78996f32ac0
Ltest

# ╔═╡ 1841a90c-521a-43b3-8bb3-470a836ec142
Qtemp1[:, 1]'

# ╔═╡ d913add1-a56d-4896-a8b4-390552de90d6
function testhess()
	EVs = zeros(h11)
	fs = zeros(h11)
	NewQ = inv(canK)' * Qtemp1
	for i in 1:h11
		fs[i] = NewQ[:, i]' * NewQ[:, i]
		EVs[i] = Ltest[i] * fs[i]
		T = CYAxiverse.generate.orth_basis(NewQ[:, i])'
		NewQ = T * NewQ
	end
	fs, EVs
end

# ╔═╡ 729c4d18-b776-4f59-bea6-ceb3b146bb89
ones(h11)' * Qtemp1

# ╔═╡ ad031cdc-9270-4ac8-a4e4-265be15b0033
function hess_test(x::Vector, Q::Matrix)
	hessian = zeros(size(Q,1), size(Q,1), size(Q, 2))
	for i in axes(Q, 2), j in axes(Q, 2)
		hessian[:, :, i] = (@view(Q[:, i]) * @view(Q[:, j])') * cos.(x' * Q)[i]
	end
	hessian
	# Hermitian(hessian + hessian' - Diagonal(hessian))
end	

# ╔═╡ 819972ff-5978-408a-a6a2-635abee2bd9d
md"""
- add sum and \Lambda to `hess_test`
"""

# ╔═╡ 37997fdb-9bee-455f-98f4-cda269136c9f
@time begin
	hessian_fill = zeros(size(Qtemp1, 1), size(Qtemp1, 1))
	hess = hess_test(zeros(h11), inv(canK)' * Qtemp1)
	for i in axes(hess, 1), j in axes(hess, 2)
		if i>=j
			hessian_fill[i, j] = sum(Ltest .* hess[i, j, :])
		end
	end
	# hessian_fill = Hermitian(hessian_fill + hessian_fill' - Diagonal(hessian_fill))
end
		

# ╔═╡ 9f1a0942-405a-4e3d-8b3a-3848a7897b50
sort(eigen(hessian_fill).values, by=x->abs(x)), sort(testhess()[2], by=x->abs(x)), sort(eigen(test_hess).values, by=x->abs(x))

# ╔═╡ 2baf651e-37aa-40fc-b1e0-1808490aed33
Ktest

# ╔═╡ 1064a293-e54f-48de-b999-993d33d67990
eigen(hessian_fill, Ktest[1:h11, 1:h11]).values

# ╔═╡ 06ef508c-3ee1-488a-aff7-38bfef9d7dc6
hess_test(zeros(h11), Qtemp1)

# ╔═╡ 83f24e49-72b4-459c-9ee4-6d16bb57d834
@time CYAxiverse.generate.hessian_norm(zeros(h11), Qtemp1) == hess_test(zeros(h11), Qtemp1)

# ╔═╡ 33f63609-dbfe-42a7-b9cc-4054ea69bd85
cholesky(Ktest[1:10, 1:10]).L

# ╔═╡ f763a26a-87ba-4695-be58-5a3ad8710da5
10^(1e-2)

# ╔═╡ e9f9f4fe-b003-4ba6-92bb-ee050deeca15
CYAxiverse.generate.hp_spectrum(10, 10, 1)["m"] .- CYAxiverse.generate.pq_spectrum(10, 10, 1)[1].m

# ╔═╡ b8bd3634-e977-4e73-8005-4d73333fdcc4
md"""
- optimise `hessian_norm` function as above
- check to see if multiplying by Λ as last step leaves eigenvalues unchanged
- use `gauss_log` function to incorporate Λ
"""

# ╔═╡ Cell order:
# ╟─fee399f9-2668-41e0-a296-37b348a04769
# ╟─90f44877-6310-49b1-9331-f8601918e4b3
# ╟─915e345e-7002-489c-8fec-8395381f0fe5
# ╟─2000a078-38f5-4c93-8627-ba6b4970aef6
# ╟─7c8e7502-94d8-4da6-a5e2-b950b33a62c2
# ╠═fbb69bcb-64c6-42c2-8ce1-666f397eb40e
# ╠═e556408b-25f7-4fae-ba0b-243242279ba8
# ╠═3788df6d-c756-4b6a-8d75-8cd018ab2991
# ╠═b1362f7d-55e5-48d6-a695-f3bf59d8bf99
# ╠═8778a5d2-5eae-426b-bc86-c62c9326c9fd
# ╟─8c7bb44d-edb3-46b3-aeef-ac21d2ee16f5
# ╠═bfd0a313-a144-4883-abee-e74ac7f4a8e4
# ╠═1e3c5202-df60-47c5-84df-31fee243fe80
# ╠═b07db615-e3c0-4d60-ab3f-13f35b99f1c9
# ╠═eb35a641-b828-42fe-b01e-ecd6ccc7212f
# ╠═2b3a2af1-091b-47df-bc2f-082b00a4342b
# ╠═4476d3fe-db5b-4ed5-aaad-5915b2eb1605
# ╠═8ff2f070-1ae8-4c0f-b540-34e76fe0d685
# ╠═4e965d07-15dd-4840-b034-2a997a935567
# ╠═d98e8576-8bf0-4a59-ad29-80b4ddba366e
# ╠═f915ce5a-56b9-4634-967c-b955c2246419
# ╠═8dd2c1c6-620d-431f-9ece-be204bbab7eb
# ╠═a6c24db6-367a-4a2a-9eed-47c00101a099
# ╠═f35bfd5e-015f-49e8-a87d-e00320785fee
# ╠═5daa448d-d9e7-4d43-b68e-b78996f32ac0
# ╠═1841a90c-521a-43b3-8bb3-470a836ec142
# ╠═d913add1-a56d-4896-a8b4-390552de90d6
# ╠═729c4d18-b776-4f59-bea6-ceb3b146bb89
# ╠═ad031cdc-9270-4ac8-a4e4-265be15b0033
# ╟─819972ff-5978-408a-a6a2-635abee2bd9d
# ╠═37997fdb-9bee-455f-98f4-cda269136c9f
# ╠═9f1a0942-405a-4e3d-8b3a-3848a7897b50
# ╠═2baf651e-37aa-40fc-b1e0-1808490aed33
# ╠═1064a293-e54f-48de-b999-993d33d67990
# ╠═06ef508c-3ee1-488a-aff7-38bfef9d7dc6
# ╠═83f24e49-72b4-459c-9ee4-6d16bb57d834
# ╠═33f63609-dbfe-42a7-b9cc-4054ea69bd85
# ╠═f763a26a-87ba-4695-be58-5a3ad8710da5
# ╠═e9f9f4fe-b003-4ba6-92bb-ee050deeca15
# ╟─b8bd3634-e977-4e73-8005-4d73333fdcc4
