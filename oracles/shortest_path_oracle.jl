#=
Here an optimization oracle is a linear optimization oracle for the set S, i.e., it solves
    z^\ast(c) :=    min c^Tw
                    s.t. w \in S
The solution of the above problem returned by the oracle is referred to as w^\ast(\cdot) in
the paper

A specific implementation of an optimization oracle should take a single input argument c,
and should return the pair (z^\ast(c), w^\ast(c))

This file contains the code for the shortest path oracle, which relies on JuMP
=#

using Gurobi, JuMP, Clp, LightGraphs, SparseArrays, LinearAlgebra

"""
    sp_flow_jump_setup(sources, destinations, start_node, end_node; solver=:Gurobi, gurobiEnv = Gurobi.Env())

Sets up a shortest path oracle using JuMP. Sets up constraints once then changes only the
objective coefficients dynamically. Takes as input a directed graph via a list of edges in the format of
(sources[i], destinations[i]) for i = 1,...,num_edges. Nodes are assumed to be integers in the range
1,...,num_nodes.


# Arguments
- `sources`:  list of sources for each edge, length of this vector is number of edges
- `destinations`:  list of destinations of each edge, length of this vector is number of edges
- `start_node`:  start node for the shortest path problem
- `end_node`:  end node for the shortest path problem
- `solver`:  either `:Gurobi` or `:Clp`
- `gurobiEnv`:  optionally pass in an existing Gurobi enviornemnet
"""
function sp_flow_jump_setup(sources, destinations, start_node, end_node; solver=:Gurobi, gurobiEnv = Gurobi.Env(), small_coefficient_tolerance = 0.01)
    nodes = unique(union(sources, destinations))
    n_nodes = length(nodes)
    n_edges = length(sources)

    # Need to hard code incidence matrix!
    # A_mat = zeros(n_nodes, n_edges)
    # for i = 1:n_edges
    #     A_mat[sources[i], i] = -1
    #     A_mat[destinations[i], i] = 1
    # end

    # Hard code sparse node-edge incidence matrix!
    I_vec = [sources; destinations]
    J_vec = [collect(1:n_edges); collect(1:n_edges)]
    V_vec = [-ones(n_edges); ones(n_edges)]
    A_mat = sparse(I_vec, J_vec, V_vec)

    # Set up RHS
    b_vec = sparse(zeros(n_nodes))
    b_vec[start_node] = -1
    b_vec[end_node] = 1

    # Remove sparse
    A_mat = round.(Int, Matrix(A_mat))
    b_vec = round.(Int, Vector(b_vec))

    # Set up JuMP model
    if solver == :Gurobi
        mod = Model(()->Gurobi.Optimizer(gurobiEnv))
    elseif solver == :Clp
        mod = Model(Clp.Optimizer)
    else
        error("Not a valid solver, either :Clp or :Gurobi")
    end

    @variable(mod, 0 <= w[1:(n_edges)] <= 1) # bound constraints
    @constraint(mod, A_mat*w .== b_vec)

    # Set up local oracle function
    function local_sp_oracle_jump(c::Vector{Float64})
        d = length(c)
        for i = 1:d
            if abs(c[i]) < small_coefficient_tolerance
                c[i] = 0
            end
        end

        @objective(mod, Min, dot(c, w))
        optimize!(mod)
        z_ast = objective_value(mod)
        w_ast = value.(w)

        return (z_ast, w_ast)
    end

    return c -> local_sp_oracle_jump(c)
end

"""
    convert_grid_to_list(dim1, dim2)

Utility function to convert a dim1 x dim2 grid graph into a (sources[i], detinations[i])
format.
"""
function convert_grid_to_list(dim1, dim2)
    g = Grid([dim1, dim2])
    sources = Int64[]
    destinations = Int64[]

    for e in edges(g)
        push!(sources, src(e))
        push!(destinations, dst(e))

        # Make it directed by commenting out the below
        #push!(sources, dst(e))
        #push!(destinations, src(e))
    end

    return sources, destinations
end
