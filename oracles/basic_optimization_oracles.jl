#=
An optimization oracle is a linear optimization oracle for the set S, i.e., it solves
    z^\ast(c) :=    min c^Tw
                    s.t. w \in S
The solution of the above problem returned by the oracle is referred to as w^\ast(\cdot) in
the paper

A specific implementation of an optimization oracle should take a single input argument c,
and should return the pair (z^\ast(c), w^\ast(c))

Below are several basic examples
=#

"""
    simplex_oracle(c)

Optimization oracle when S is the unit simplex
"""
function simplex_oracle(c::Vector{Float64})
    (z, ind) = findmin(c)
    w = zeros(length(c))
    w[ind] = 1
    return (z, w)
end

"""
    svm_oracle(c)

Optimization oracle for the SVM example where S = [-1/2, +1/2].

# Arguments
- `c` is assumed to be a vector of length 1, NOT a scalar
"""
function svm_oracle(c::Vector{Float64})
    c_val = c[1]
    if c_val > 0
        w = [-0.5]
        z = (-0.5)*c_val
    else
        w = [0.5]
        z = 0.5*c_val
    end

    return (z, w)
end
