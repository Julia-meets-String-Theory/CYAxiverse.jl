### A Pluto.jl notebook ###
# v0.19.13

using Markdown
using InteractiveUtils

# ╔═╡ daa0a007-2898-4c9b-9f35-b296574cdb41
using Pkg; 

# ╔═╡ 09ef6adb-c76a-469d-8f25-3d2089a6f684
Pkg.activate("/scratch/users/mehta2/cyaxiverse/CYAxiverse");

# ╔═╡ 0227789b-1548-4d6a-ad2f-ef017530849b
using PlutoUI, HDF5, ArbNumerics, LineSearches, Optim, CairoMakie, Distributions, LinearAlgebra, ProgressLogging, Revise, Random, SparseArrays, LeftChildRightSiblingTrees


# ╔═╡ ce668813-d11a-4dc0-8f0b-3c3dcdd039f3
using CYAxiverse

# ╔═╡ 69dea65d-54ce-4abd-8476-6fe481cdff57
md"""
# Vacua search -- Numerics
"""

# ╔═╡ 490d106e-9866-4f6e-8b06-db4de0309576
md"""
### This notebook is for tree testing
"""

# ╔═╡ 6e6fc4aa-7fcb-11ed-1ba1-756d821506e6
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

# ╔═╡ 344896d8-52bd-4cf6-834d-86fd3e7e9c82
begin
	ENV["newARGS"] = string("vacua_stretchtest")
	md"""
	The database we are using is $(ENV["newARGS"])
	"""
end

# ╔═╡ 668b0b78-a680-4e84-be29-f64471c7163c
CYAxiverse.filestructure.present_dir()


# ╔═╡ 87be2229-742b-4299-8061-69a1a7398940
begin
	h11list = CYAxiverse.filestructure.paths_cy()[2]
	md"""
	There are a total of $(size(h11list,2)) geometries in the database with $(minimum(Set(h11list[1,:]))) ≤ ``h^{1,1}`` ≤ $(maximum(Set(h11list[1,:])))
	"""
end

# ╔═╡ c6e264b3-5ea2-4028-a26f-c40ab4dbf2b1
begin
	min0 = Node(0.)
	min1 = addsibling(min0, 2π/3)
	min2 = addsibling(min1, 4π/3)
	min00 = addchild(min0, 0.)
	min01 = addsibling(min00, 2π/5 - min1.data)
	min10 = addchild(min1, 0.)
	min11 = addsibling(min10, 2π/8)
	min20 = addchild(min2, 0.)
	min21 = addsibling(min20, 2π/6)
end

# ╔═╡ d5c0150a-916b-47c4-aa80-86d2a0c6bd73
:a[1:10]

# ╔═╡ 55b5c42f-714e-44c8-964d-d2d605f43211
function showtree(node, indent=0)
   println("\t"^indent, node.data)
   for child in node
	   showtree(child, indent + 1)
   end
end

# ╔═╡ 8402ea92-3e90-4709-be8c-9edd99504cf0
LeftChildRightSiblingTrees.showedges(min0)

# ╔═╡ b7c5dd3d-b9f8-4470-a2b1-58bf290ee170
collect(CYAxiverse.structs.ParentTrack(t))

# ╔═╡ b15c87a6-c028-4ad6-83bc-23ac7840c260
[CYAxiverse.structs.MyTree(i, ts[i]) for i ∈ axes(ts,1)]

# ╔═╡ 47473bda-1619-499a-96ba-cb44e9b28a55
t.subtrees[3].subtrees[1].data

# ╔═╡ 2edfe043-9a4a-4e64-a776-4d329ff05b91
lqhat = CYAxiverse.generate.:αmatrix(CYAxiverse.structs.GeometryIndex(h11=10,polytope=10,frst=1))

# ╔═╡ 9620b336-cbd8-4e64-b3f0-2790bd7fbcb2


# ╔═╡ 4e282b9f-9c58-4d46-a2a7-ab6ed21c87e0
CYAxiverse.generate.omega(lqhat.Qhat).Ωparallel

# ╔═╡ e3ceb403-cb73-4195-a717-ad7f7f2d14a9
begin
	Ωperp = CYAxiverse.generate.omega(lqhat.Qhat).Ωperp
	[max(2π / norm(Ωperp[:,i]), 2π) for i in axes(Ωperp, 2)]
end

# ╔═╡ 04222ee1-0f18-4b41-9900-b19ef9f32e57
@time begin
	i = 3
	(CYAxiverse.generate.project_out(Vector(lqhat.Qhat[:,i])).Π * lqhat.Qhat[:, i+1:end])

end

# ╔═╡ a6c45afe-c9a6-421b-8414-731e6ba0971b
@time vcat(mapslices(norm, CYAxiverse.generate.project_out(Vector(lqhat.Qhat[:,3])).Π * lqhat.Qhat[:, 3+1:end]; dims=1)', zeros(3))

# ╔═╡ 09b8c30c-3393-4d32-b89b-c2bd01d57ee1
sqrt(2//25)

# ╔═╡ Cell order:
# ╟─69dea65d-54ce-4abd-8476-6fe481cdff57
# ╟─490d106e-9866-4f6e-8b06-db4de0309576
# ╟─6e6fc4aa-7fcb-11ed-1ba1-756d821506e6
# ╟─344896d8-52bd-4cf6-834d-86fd3e7e9c82
# ╟─daa0a007-2898-4c9b-9f35-b296574cdb41
# ╟─09ef6adb-c76a-469d-8f25-3d2089a6f684
# ╠═0227789b-1548-4d6a-ad2f-ef017530849b
# ╠═ce668813-d11a-4dc0-8f0b-3c3dcdd039f3
# ╟─668b0b78-a680-4e84-be29-f64471c7163c
# ╠═87be2229-742b-4299-8061-69a1a7398940
# ╠═c6e264b3-5ea2-4028-a26f-c40ab4dbf2b1
# ╠═d5c0150a-916b-47c4-aa80-86d2a0c6bd73
# ╠═55b5c42f-714e-44c8-964d-d2d605f43211
# ╠═8402ea92-3e90-4709-be8c-9edd99504cf0
# ╠═b7c5dd3d-b9f8-4470-a2b1-58bf290ee170
# ╠═b15c87a6-c028-4ad6-83bc-23ac7840c260
# ╠═47473bda-1619-499a-96ba-cb44e9b28a55
# ╠═2edfe043-9a4a-4e64-a776-4d329ff05b91
# ╠═9620b336-cbd8-4e64-b3f0-2790bd7fbcb2
# ╠═4e282b9f-9c58-4d46-a2a7-ab6ed21c87e0
# ╠═e3ceb403-cb73-4195-a717-ad7f7f2d14a9
# ╠═04222ee1-0f18-4b41-9900-b19ef9f32e57
# ╠═a6c45afe-c9a6-421b-8414-731e6ba0971b
# ╠═09b8c30c-3393-4d32-b89b-c2bd01d57ee1
