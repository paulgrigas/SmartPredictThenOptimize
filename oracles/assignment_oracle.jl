#=
An optimization oracle is a linear optimization oracle for the set S, i.e., it solves
    z^\ast(c) :=    min c^Tw
                    s.t. w \in S
The solution of the above problem returned by the oracle is referred to as w^\ast(\cdot) in
the paper

A specific implementation of an optimization oracle should take a single input argument c,
and should return the pair (z^\ast(c), w^\ast(c))

This file contains the code for the assignment problem optimization oracle, which
relies on the Hungarian solver package.
=#

using Hungarian

"""
    assignment_oracle(c)

Optimization oracle for the assignment problem. Uses Hungarian algorithm. Assumes
that the input c is a vector of length d^2 and must be reshaped into a matrix
"""
function assignment_oracle(c::Vector{Float64})
    d_sq = length(c)
    d = isqrt(d_sq)
    C = reshape(c, (d,d))
    assignment, z = hungarian(C)

    W = zeros(d, d)
    for i = 1:d
        W[i, assignment[i]] = 1
    end
    w = vec(W)
    return (z, w)
end

"""
    assignment_hamming_reduce(c)

Utility function that converts c to a cost matrix with 0 on the optimal edges and 1 on all
other possible edges. Therefore, the SPO loss becomes equal to the hamming loss between the
predicted assignment and the optimal one. Returns the new cost vector c associated with
the Hamming loss.
"""
function assignment_hamming_reduce(c::Vector{Float64})
    d_sq = length(c)
    d = isqrt(d_sq)
    C = reshape(c, (d,d))
    assignment, z = hungarian(C)

    new_C = ones(d, d)
    for i = 1:d
        new_C[i, assignment[i]] = 0
    end
    new_c = vec(new_C)
    return new_c
end
