using LinearAlgebra

function repro_l_layout()
    qprime = [1.0 2.0; 3.0 4.0]
    tau = [0.5, 0.25]
    Kinv = [1.0 0.0; 0.0 2.0]
    V = 10.0

    L1 = zeros(size(qprime, 1), 2)
    L2 = zeros(1, 2)

    L1[1, :] = [
        (8 * pi / V^2) * dot(qprime[1, :], tau),
        -2 * log10(ℯ) * pi * dot(qprime[1, :], tau),
    ]
    L1[2, :] = [
        (8 * pi / V^2) * dot(qprime[2, :], tau),
        -2 * log10(ℯ) * pi * dot(qprime[2, :], tau),
    ]

    L2[1, :] = [
        (pi * dot(qprime[1, :], (Kinv * qprime[2, :])) +
         dot(qprime[1, :] + qprime[2, :], tau)) * 8 * pi / V^2,
        -2 * log10(ℯ) * pi * (dot(qprime[1, :], tau) + dot(qprime[2, :], tau)),
    ]

    L = vcat(L1, L2)

    println("L shape: ", size(L))
    println("L rows:")
    display(L)

    @assert size(L, 2) == 2
    @assert all(L[i, 1] != L[i, 2] for i in axes(L, 1))

    return L
end

repro_l_layout()
