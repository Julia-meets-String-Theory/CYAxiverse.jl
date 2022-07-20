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

function vacua_profiler(L::Matrix{Float64},Q::Matrix{Int})
    reset_timer!()
    @timeit "h11" h11::Int = size(Q,2)
    
    ###### Nemo SNF #####
    # @timeit "Nemo matrix" Qtemp::Nemo.fmpz_mat = matrix(Nemo.ZZ,Q)
    # @timeit "SNF" T::Nemo.fmpz_mat = snf_with_transform(Qtemp)[2]
    # @timeit "inv(T)" Tparallel1::Nemo.fmpz_mat = inv(T)[:,1:h11]
    # @timeit "convert T∥" Tparallel::Matrix{Int} = convert(Matrix{Int},Tparallel1)

    ###### wildart SNF #####
    @timeit "SNF" F = smith(Q)
    @timeit "T" T::Matrix{Int} = F.S
    @timeit "inv(T)" Tparallel::Matrix{Int} = round.(inv(T)[:,1:h11])
    # println(size(T))
    
    @timeit "θparallel" θparalleltest::Matrix{Float64} = inv(transpose(Float64.(Q)) * Float64.(Q)) * transpose(Float64.(Q)) * Float64.(Tparallel)
    @timeit "zip LQ" LQtest::Matrix{Float64} = hcat(L,Q);
    @timeit "sort LQ" LQsorted::Matrix{Float64} = LQtest[sortperm(L[:,2], rev=true), :]
    @timeit "unzip LQ" Lsorted_test::Matrix{Float64},Qsorted_test::Matrix{Int} = LQsorted[:,1:2], Int.(LQsorted[:,3:end])
    @timeit "init Qtilde" Qtilde::Matrix{Int} = hcat(zeros(Int,size(Qsorted_test[1,:],1)),Qsorted_test[1,:])
    for i=2:size(Qsorted_test,1)
        @timeit "Matrix.Space" S::Nemo.FmpzMatSpace = MatrixSpace(Nemo.ZZ, size(Qtilde,1), (size(Qtilde,2)))
        @timeit "lin. ind." m::Nemo.fmpz_mat = S(hcat(Qtilde[:,2:end],Qsorted_test[i,:]))
        @timeit "NullSpace" d::Int = Nemo.nullspace(m)[1]
        if d == 0
            @timeit "Qtilde" Qtilde = hcat(Qtilde,Qsorted_test[i,:])
        end
    end
    @timeit "vacua" vacua::Int = round(abs(det(θparalleltest) / det(inv(Float64.(Qtilde[:,2:end])))))
    # if h11 <= 50
    #     @timeit "thparallel - Rational" thparallel::Matrix{Rational} = Rational.(round.(θparalleltest; digits=5))
    #     @timeit "make Dict" keys = ["vacua","θ∥","Qtilde"]
    #     vals = [abs(vacua), thparallel, Qtilde[:,2:end]]
    #     print_timer()
    #     return Dict(zip(keys,vals))
    # else
    keys = ["vacua","θ∥","Qtilde"]
    vals = [abs(vacua), θparalleltest, Qtilde[:,2:end]]
    print_timer()
    return Dict(zip(keys,vals))
    # end
end

end