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
		margin: 0 auto;
		max-width: 98%;
    	padding-left: max(160px, 5%);
    	padding-right: max(160px, 5%);
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

# ╔═╡ 1ef56dc0-dce2-4f76-9632-13209fc7a0ba
typeof(CYAxiverse.read.vacua_jlm(CYAxiverse.structs.GeometryIndex(5,10,1))) <: CYAxiverse.structs.Min_JLM_1D

# ╔═╡ eb3a354d-dedf-4b7d-ab14-d8474f217ad4
CYAxiverse.generate.LQtildebar(9, 40, 1)

# ╔═╡ b26c101b-0387-4d32-a5ed-bc0d48871dcf
CYAxiverse.generate.LQtilde(9, 40, 1).Qtilde == CYAxiverse.generate.LQtildebar(9, 40, 1)["Qhat"]

# ╔═╡ 52e925f4-7c98-45a6-ba17-02c7d58e3ef9
@time begin
	geom_idx = CYAxiverse.structs.GeometryIndex(8, 40, 1)
	αtest = CYAxiverse.generate.:αmatrix(geom_idx)
	pq_spec_test = CYAxiverse.generate.pq_spectrum(geom_idx)
	hp_spec_test = CYAxiverse.generate.hp_spectrum(geom_idx)
	hp_spec_test["m"], pq_spec_test[1].m
end

# ╔═╡ c6b70020-b7bd-438d-bc9f-c7df05249e71


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

# ╔═╡ 92fdaa86-1c40-4a18-acb4-26756a98b31a
log10.(sqrt.(eigen(CYAxiverse.generate.hessian(zeros(size(Qtest, 1)), hcat(Lsign, Ltest), Matrix(Qtest)), Ktest).values)) .+ Float64(log10(CYAxiverse.generate.constants()["MPlanck"])) .+9 .+ Float64(CYAxiverse.generate.constants()["log2π"])

# ╔═╡ ef215872-d011-4432-bee3-8460c82f45e7
@time CYAxiverse.generate.hp_spectrum(Ktest, hcat(Lsign, Ltest), Matrix{Int}(Qtest'); prec = 200)["m"], CYAxiverse.generate.pq_spectrum(Ktest, hcat(Lsign, Ltest), Matrix{Int}(Qtest'))[1].m

# ╔═╡ 860209f4-1ae4-498b-ac75-c5fa94c2608c
Vector(Qtest[1, :])

# ╔═╡ 5d0447c8-3f1f-4c7c-ad13-13965251a5fd
CYAxiverse.generate.orth_basis(Vector(Qtest[1, :])) * Qtest

# ╔═╡ eb35a641-b828-42fe-b01e-ecd6ccc7212f
CYAxiverse.generate.pq_spectrum(Ktest, hcat(Lsign, Ltest), Matrix{Int}(Qtest'))

# ╔═╡ dac736b9-8bba-4603-af8c-b250112d5db1
cholesky(Ktest).L

# ╔═╡ ee17361e-dedd-42bd-b9e9-e84791bad7e5
LowerTriangular(qr(Ktest).Q)

# ╔═╡ Cell order:
# ╟─fee399f9-2668-41e0-a296-37b348a04769
# ╟─90f44877-6310-49b1-9331-f8601918e4b3
# ╟─915e345e-7002-489c-8fec-8395381f0fe5
# ╟─2000a078-38f5-4c93-8627-ba6b4970aef6
# ╠═7c8e7502-94d8-4da6-a5e2-b950b33a62c2
# ╠═fbb69bcb-64c6-42c2-8ce1-666f397eb40e
# ╠═e556408b-25f7-4fae-ba0b-243242279ba8
# ╠═3788df6d-c756-4b6a-8d75-8cd018ab2991
# ╠═b1362f7d-55e5-48d6-a695-f3bf59d8bf99
# ╠═8778a5d2-5eae-426b-bc86-c62c9326c9fd
# ╟─8c7bb44d-edb3-46b3-aeef-ac21d2ee16f5
# ╠═1ef56dc0-dce2-4f76-9632-13209fc7a0ba
# ╠═eb3a354d-dedf-4b7d-ab14-d8474f217ad4
# ╠═b26c101b-0387-4d32-a5ed-bc0d48871dcf
# ╠═52e925f4-7c98-45a6-ba17-02c7d58e3ef9
# ╠═c6b70020-b7bd-438d-bc9f-c7df05249e71
# ╠═bfd0a313-a144-4883-abee-e74ac7f4a8e4
# ╠═b07db615-e3c0-4d60-ab3f-13f35b99f1c9
# ╠═92fdaa86-1c40-4a18-acb4-26756a98b31a
# ╠═ef215872-d011-4432-bee3-8460c82f45e7
# ╠═860209f4-1ae4-498b-ac75-c5fa94c2608c
# ╠═5d0447c8-3f1f-4c7c-ad13-13965251a5fd
# ╠═eb35a641-b828-42fe-b01e-ecd6ccc7212f
# ╠═dac736b9-8bba-4603-af8c-b250112d5db1
# ╠═ee17361e-dedd-42bd-b9e9-e84791bad7e5
