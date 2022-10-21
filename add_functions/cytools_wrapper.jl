#############################################################################
####### !!!!!! If this file stops the package compiling !!!!!! ##############
### !! First go into singularity / docker image and run julia !! ############
##### !! then check ENV["PYTHON"] is set correctly !! #######################
###### !! (if not, need to exit julia and export python path when !! ########
######## !! loading singularity / docker image) !! ##########################
##### !! then Pkg.build("PyCall") -- this will then build it with !! ########
##### !! the correct python installation, i.e. the one in the image !! ######
##### !! now the package should recompile !! ################################
#############################################################################

"""
    CYAxiverse.cytools_wrapper
Functions that wrap basic functionality of [CYTools](https://cytools.liammcallistergroup.com/) in order to pull polytopes from the [Kreuzer-Skarke Database](https://doi.org/10.4310/ATMP.2000.v4.n6.a2)

"""
module cytools_wrapper

using ..filestructure: cyax_file, present_dir, np_path_generate
using ..read: topology

using PyCall
using HDF5
using LinearAlgebra

"""
    __init__()
Here we initialise the CYTools functions 
(for further details and argument info see https://cytools.liammcallistergroup.com/docs/documentation/):

test_config() -- this function checks that the mosek_license file is found

f_polytopes(h11,h12,h13,h21,h22,h31,chi,lattice,dim,n_points,n_vertices,
        n_dual_points,n_facets,limit,timeout,as_list,backend, dualize,favorable) -- 
            this function pulls polytopes from the online KS database, i.e. calls fetch_polytopes
            from CYTools

poly(points, backend) -- this function allows access to the PyObject Polytope
"""
function __init__()
    py"""
    from cytools import config
    import os
    config.set_mosek_path(os.environ['HOME'])
    config.check_mosek_license()
    def test_config():
       return config.mosek_is_activated
    """

    py"""
    import numpy as np
    import scipy as sp
    from cytools import fetch_polytopes
    from cytools import Polytope
    def f_polytopes(h11=None, h12=None, h13=None, h21=None, h22=None, h31=None,
                    chi=None, lattice=None, dim=4, n_points=None,
                    n_vertices=None, n_dual_points=None, n_facets=None,
                    limit=1000, timeout=60, as_list=False, backend=None,
                    dualize=False, favorable=None):
        return fetch_polytopes(h11,h12,h13,h21,h22,h31,chi,lattice,dim,n_points,n_vertices,
        n_dual_points,n_facets,limit,timeout,as_list,backend, dualize,favorable)

    def poly(points, backend=None):
        return Polytope(points,backend)
    """
end

fetch_polytopes(h11,limit; lattice="N",as_list=false,favorable=false) = py"f_polytopes(h11=$h11,limit=$limit, lattice=$lattice, as_list=$as_list, favorable=$favorable)"

poly(points; backend=nothing) = py"poly($points, backend=$backend)"



function topologies(h11,n)
    h11list_temp = []
    tri_test = []
    tri_test_m = []
    #Generate list of $n polytopes at $h11
    poly_test = fetch_polytopes(h11,4*n, lattice="N", as_list=true, favorable=true)
    #Locator for points of polytope for saving
    points = [p.points() for p in poly_test]
    #If number of polytopes < $n, generate more triangulations per polytope, 
    #otherwise generate 1 triangulation per polytope upto $n
    spt = size(poly_test,1)
    m = nothing;
    if spt == 0
        return [0, 0, 0, 0]
    elseif spt < n && h11 > 3
        left_over = mod(n, spt)
        m = n ÷ spt
        if left_over == 0
            tri_test_m = [poly_test[i].random_triangulations_fast(N=m, as_list=true, progress_bar=false) for i=1:spt];
            cy_num = [size(tri_test_m[i],1) for i=1:size(tri_test_m,1)]
            tri_test = vcat(tri_test_m...)
        else
            tri_test_m = [poly_test[i].random_triangulations_fast(N=m, as_list=true, progress_bar=false) for i=left_over+1:spt];
            tri_test_m1 = [poly_test[i].random_triangulations_fast(N=m+1, as_list=true, progress_bar=false) for i=1:left_over];
            tri_test_m = vcat(tri_test_m1, tri_test_m)
            cy_num = [size(tri_test_m[i],1) for i=1:size(tri_test_m,1)]
            cy_num1 = [size(tri_test_m1[i],1) for i=1:size(tri_test_m1,1)]
            cy_num = vcat(cy_num1,cy_num)
            tri_test = vcat(tri_test_m1...,tri_test_m...)
        end
    else
        tri_test = [poly_test[i].triangulate() for i=1:n];
        points = points[1:n]
    end
    simplices = []
    cy = []
    for t=1:size(tri_test,1)
        #Locator for simplices of triangulations for saving
        push!(simplices,tri_test[t].simplices())
        #Generate list of CY3s
        push!(cy,tri_test[t].get_cy())
    end
        #Create dir for saving -- structure is h11_{$h11}.zfill(3)/np_{$tri}.zfill(7)/cy_{$cy}.zfill(3)/data
    if isdir(string(present_dir(),"h11_",lpad(h11,3,"0")))
    else
        mkdir(string(present_dir(),"h11_",lpad(h11,3,"0")))
    end
    if m==nothing
        for tri=1:size(tri_test,1)
            if isdir(string(present_dir(),"h11_",lpad(h11,3,"0"),"/np_",lpad(tri,7,"0")))
            else
                mkdir(string(present_dir(),"h11_",lpad(h11,3,"0"),"/np_",lpad(tri,7,"0")))
            end
            if isdir(string(present_dir(),"h11_",lpad(h11,3,"0"),"/np_",lpad(tri,7,"0"),"/cy_",lpad(1,7,"0")))
            else
                mkdir(string(present_dir(),"h11_",lpad(h11,3,"0"),"/np_",lpad(tri,7,"0"),"/cy_",lpad(1,7,"0")))
            end
            if isfile(cyax_file(h11,tri,1))
                rm(cyax_file(h11,tri,1))
            end
            h5open(cyax_file(h11,tri,1), "cw") do file
                f1 = create_group(file, "cytools")
                f1a = create_group(f1, "geometric")
                f1a["points",deflate=9] = Int.(points[tri])
                f1a["simplices",deflate=9] = Int.(simplices[tri])
            end
            push!(h11list_temp, [h11,cy[tri],tri,1])
        end
    else
        n = 1
        for tri=1:size(tri_test_m,1)
            for cy_i=1:size(tri_test_m[tri],1)
                if isdir(string(present_dir(),"h11_",lpad(h11,3,"0"),"/np_",lpad(tri,7,"0")))
                else
                    mkdir(string(present_dir(),"h11_",lpad(h11,3,"0"),"/np_",lpad(tri,7,"0")))
                end
                if isdir(string(present_dir(),"h11_",lpad(h11,3,"0"),"/np_",lpad(tri,7,"0"),"/cy_",lpad(cy_i,7,"0")))
                else
                    mkdir(string(present_dir(),"h11_",lpad(h11,3,"0"),"/np_",lpad(tri,7,"0"),"/cy_",lpad(cy_i,7,"0")))
                end
                if isfile(cyax_file(h11,tri,cy_i))
                    rm(cyax_file(h11,tri,cy_i))
                end
                h5open(cyax_file(h11,tri,cy_i), "cw") do file
                    f1 = create_group(file, "cytools")
                    f1a = create_group(f1, "geometric")
                    f1a["points",deflate=9] = Int.(points[tri])
                    f1a["simplices",deflate=9] = Int.(simplices[n])
                end
                push!(h11list_temp, [h11,cy[n],tri,cy_i])
                n+=1
            end
        end
    end
    h11list = hcat(h11list_temp...)
    GC.gc()
    return h11list
end
function cy_from_poly(h11)
    h11list_temp = []
    h11list_inds = np_path_generate(h11)
    for col in eachcol(h11list_inds)
        h11,tri,cy_i = col
        top_data = topology(h11,tri,cy_i)
        points, simplices = top_data["points"], top_data["simplices"]
        p = poly(points)
        t = p.triangulate(simplices=simplices)
        cy = t.get_cy()
        push!(h11list_temp,[h11,cy,tri,cy_i])
    end
    h11list = hcat(h11list_temp...)
    GC.gc()
    return h11list
end

function geometries(h11,cy,tri,cy_i=1)
    glsm = zeros(Int,h11,h11+4)
    basis = zeros(Int,h11)
    tip = zeros(Float64,h11)
    Kinv = zeros(Float64,h11,h11)
    K = zeros(Float64,h11,h11)
    tau = zeros(Float64,h11)
    qprime = zeros(Int,h11+4,h11)
    #Locator for h21s for saving
    h21::Int = cy.h21()
    #GLSM basis for saving
    glsm = cy.glsm_charge_matrix(include_origin=false)
    #Divisor basis for saving (allows for reproducibility)
    basis = cy.divisor_basis()
    #Find tip of SKC
    n,m = 1,1
    tip = cy.toric_kahler_cone().tip_of_stretched_cone(sqrt(n))
    #Kinv at tip -- save this or save K?
    Kinv = cy.compute_Kinv(tip)
    Kinv = Hermitian(1/2 * Kinv + Kinv')
    #Generate list of Q matrices -- only $h11+4 directions
    qprime = cy.toric_effective_cone().rays()
    #PTD volumes at tip
    tau = cy.compute_divisor_volumes(tip)[basis]
    while true
        rhs_constraint = zeros(size(qprime,1))
        lhs_constraint = zeros(size(qprime,1),size(qprime,1))
        for i=1:size(qprime,1)
            for j=1:size(qprime,1)
                if i>j
                    lhs_constraint[i,j] = abs.(log.(abs.(pi*dot(qprime[i,:],(Kinv * qprime[j,:])))) .+ (-2π * dot(tau, qprime[i,:] .+ qprime[j,:])))
                end
            end
            rhs_constraint[i] = abs.(log.(abs.(dot(tau, qprime[i, :]))) .+ (-2π * dot(tau, qprime[i,:])))
        end
        if LowerTriangular(lhs_constraint .< rhs_constraint) - I(h11+4) == LowerTriangular(zeros(h11+4, h11+4))
            break
        else
            m+=1e-2
            tip = m .* tip
            #PTD volumes at tip
            tau = cy.compute_divisor_volumes(tip)[basis]
            #Kinv at tip -- save this or save K?
            Kinv = cy.compute_Kinv(tip)
            Kinv = Hermitian(1/2 * Kinv + Kinv') 
        end
    end
    if (minimum(tau) > 1.)
    else
        n = 1. / minimum(tau)
        tip = sqrt(n) .* tip
        #PTD volumes at tip
        tau = cy.compute_divisor_volumes(tip)[basis]
        #Kinv at tip -- save this or save K?
        Kinv = cy.compute_Kinv(tip)
        Kinv = Hermitian(1/2 * Kinv + Kinv')
    end
    tip_prefactor = [sqrt(n),m]
    #Volume of CY3 at tip
    V = cy.compute_cy_volume(tip)

    q = zeros(Int,h11+4+binomial(h11+4,2),h11)
    L2 = zeros(Float64,binomial(h11+4,2),2)
    n=1
    q[1:h11+4,:] = qprime
    for i=1:size(qprime,1)-1
        for j=i+1:size(qprime,1)
            q[h11+4+n,:] = qprime[i,:]-qprime[j,:]
            L2[n,:] = [(pi*dot(qprime[i,:],(Kinv * qprime[j,:])) 
                    + dot((qprime[i,:]+qprime[j,:]),tau))*8*pi/V^2 
                    -2*log10(exp(1))*pi*(dot(qprime[i,:],tau)+ dot(qprime[j,:],tau))]
            n+=1
        end
    end
    #Use scalar potential eqn to generate \Lambda^4 (this produces a (h11+4,2) matrix 
    #where the components are in (mantissa, exponent)(base 10) format
    #L1 are basis instantons and L2 are cross terms
    L1 = zeros(h11+4,2)
    for j=1:size(qprime,1)
        L1[j,:] = [(8*pi/V^2)*dot(qprime[j,:],tau) -2*log10(exp(1))*pi*dot(qprime[j,:],tau)]
    end
    #concatenate L1 and L2
    L = zeros(Float64,h11+4+binomial(h11+4,2),2)
    L = vcat(L1,L2)

    h5open(cyax_file(h11,tri,cy_i), "r+") do file
        if haskey(file, "cytools/geometric/h21")
        else
            file["cytools/geometric/h21",deflate=9] = h21
            file["cytools/geometric/glsm",deflate=9] = Int.(glsm)
            file["cytools/geometric/basis",deflate=9] = Int.(basis)
            file["cytools/geometric/tip",deflate=9] = Float64.(tip)
            file["cytools/geometric/tip_prefactor",deflate=9] = Float64.(tip_prefactor)
            file["cytools/geometric/CY_volume",deflate=9] = Float64(V)
            file["cytools/geometric/divisor_volumes",deflate=9] = Float64.(tau)
            file["cytools/geometric/Kinv",deflate=9] = Float64.(Kinv)
        end
        if haskey(file, "cytools/potential")
        else
            f1b = create_group(file, "cytools/potential")
            f1b["L",deflate=9] = hcat(sign.(L[:,1]), log10.(abs.(L[:,1])) .+ L[:,2])
            f1b["Q",deflate=9] = Int.(q)
        end
    end
    GC.gc()
    # return [h11,tri,cy_i]
end

end