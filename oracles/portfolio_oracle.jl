#=
An optimization oracle is a linear optimization oracle for the set S, i.e., it solves
    z^\ast(c) :=    min c^Tw
                    s.t. w \in S
The solution of the above problem returned by the oracle is referred to as w^\ast(\cdot) in
the paper

A specific implementation of an optimization oracle should take a single input argument c,
and should return the pair (z^\ast(c), w^\ast(c))

This file contains the code for the portfolio problem optimization oracle, which
relies on the JuMP and Gurobi packages.
=#

using JuMP, Gurobi, LinearAlgebra

"""
    portfolio_simplex_oracle_jump_basic(c, Sigma, gamma)

Optimization oracle in JuMP for the portfolio problem:
    min  c^T w
    s.t. w^T Sigma w <= gamma
         e^T w <= 1
         w >= 0
Here c is equal to minus the vector of returns, Sigma is the covariance matirx of the returns,
and gamma is the desired risk level in terms of the overall variance of the portfolio. Short
selling is not allowed.
"""
function portfolio_simplex_oracle_jump_basic(c::Vector{Float64}, Sigma::Matrix{Float64}, gamma::Float64)
    d = length(c)

    mod = Model(with_optimizer(Gurobi.Optimizer, OutputFlag = 0))
    @variable(mod, w[1:d] >= 0)
    @constraint(mod, sum(w[i] for i = 1:d) <= 1)
    @constraint(mod, w'*Sigma*w <= gamma)

    @objective(mod, Min, dot(c, w))

    optimize!(mod)
    z_ast = objective_value(mod)
    w_ast = value.(w)

    return (z_ast, w_ast)
end

"""
Smarter way of implementing the oracle in JuMP. Constructs the feasible region from
(Sigma, gamma) ahead of time and then returns an oracle that is just a function of c.
"""
function portfolio_simplex_jump_setup(Sigma::Matrix{Float64}, gamma::Float64; gurobiEnv = Gurobi.Env())
    (d, d2) = size(Sigma)

    if d != d2
        error("Sigma dimensions don't match")
    end

    mod = Model(with_optimizer(Gurobi.Optimizer, gurobiEnv))
    @variable(mod, w[1:d] >= 0)
    @constraint(mod, sum(w[i] for i = 1:d) <= 1)
    @constraint(mod, w'*Sigma*w <= gamma)

    function local_portfolio_oracle(c::Vector{Float64})
        @objective(mod, Min, dot(c, w))

        optimize!(mod)
        z_ast = objective_value(mod)
        w_ast = value.(w)

        return (z_ast, w_ast)
    end

    return c -> local_portfolio_oracle(c)
end
