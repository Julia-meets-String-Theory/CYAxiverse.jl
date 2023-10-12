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
	Ltest = [10. ^-i for i in 1:23]
	Lsign = ones(23)
	Qtest = Int.(I(22))
	Qtest = hcat(Qtest, zeros(22))
	Qtest[1, end] = 1
	size(Qtest), size(Ltest)
end

# ╔═╡ b07db615-e3c0-4d60-ab3f-13f35b99f1c9
@time begin
	Ktest = rand(22,22)
	Ktest = Hermitian(1/2 .* Ktest'*Ktest)
end

# ╔═╡ eb35a641-b828-42fe-b01e-ecd6ccc7212f
CYAxiverse.generate.pq_spectrum(Ktest, hcat(Lsign, Ltest), Matrix{Int}(Qtest'))

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
# ╠═b07db615-e3c0-4d60-ab3f-13f35b99f1c9
# ╠═eb35a641-b828-42fe-b01e-ecd6ccc7212f
# ╠═2b3a2af1-091b-47df-bc2f-082b00a4342b
# ╠═4476d3fe-db5b-4ed5-aaad-5915b2eb1605
# ╠═8ff2f070-1ae8-4c0f-b540-34e76fe0d685
# ╠═4e965d07-15dd-4840-b034-2a997a935567
