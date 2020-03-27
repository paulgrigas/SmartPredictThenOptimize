#=
Conic reformulation approach for the shortest path problem.
=#

using Parameters, Gurobi, JuMP, Clp, Gadfly, SparseArrays, LinearAlgebra

## Type for reformulation algorithm parameters with default values
@with_kw struct reformulation_path_parms
    lambda_max::Union{Missing, Float64} = missing
    lambda_min_ratio::Float64 = 0.0001
    num_lambda::Int = 100
    solver::Symbol = :Gurobi
    gurobiEnv::Gurobi.Env = Gurobi.Env()
    regularization::Symbol = :ridge
    regularize_first_column_B::Bool = false
    upper_bound_B_present::Bool = false
    upper_bound_B::Float64 = 10.0^6
    po_loss_function::Symbol = :leastSquares
    huber_delta::Float64 = 0.0
    verbose::Bool = false
    algorithm_type::Symbol = :fake_algorithm
end

## Type for specifying a shortest path graph instance
@with_kw struct shortest_path_graph
    sources::Array{Int,1}
    destinations::Array{Int,1}
    start_node::Int
    end_node::Int
    acyclic::Bool = false
end

"""
    sp_reformulation_path_jump(X, c, sp_graph::shortest_path_graph;
        sp_oracle = missing, reform_alg_parms::reformulation_path_parms = reformulation_path_parms())

Solves the empirical risk SPO+ problem for the specific case of the shortest path problem
with the reformulation approach using JuMP. Solves over a grid of `num_lamba` regularization
parameter values that are evenly spaced on a log scale between `lambda_min_ratio*lambda_max` and `lambda_max`.
Both Ridge and Lasso are implemented. Returns `B_soln_list, lambdas` where `B_soln_list` is a list of
lienar models such that `B_soln_list[i]` is the solution with parameter `lambdas[i]`.

The default values for the algorithm parameters `reform_alg_parms` are (explanations are given below):
    lambda_max::Number = -1
    lambda_min_ratio::Float64 = 0.0001
    num_lambda::Int = 100
    solver::Symbol = :Gurobi
    regularization::Symbol = :ridge

# Arguments
- `sources`:  list of sources for each edge, length of this vector is number of edges
- `destinations`:  list of destinations of each edge, length of this vector is number of edges
- `start_node`:  start node for the shortest path problem
- `end_node`:  end node for the shortest path problem
- `X`: p x n training set feature matrix
- `c`: d x n training set matrix of cost vectors
- `lambda_max`:  largest value of lambda, default is `(d/n)*(norm(X)^2)`
- `lambda_min_ratio`:  lambda_min is `lambda_max*lambda_min_ratio`
- `num_lambda`:  number of lambda values, evenly spaced on log scale between [lambda_min, lambda_max]
- `solver`:  either `:Gurobi` or `:Clp`
- `sp_oracle`:  optionally provide the optimization oracle corresponding to `sp_graph`
- `gurobiEnv` is a Gurobi enviornment which can be passed in (helpful to minimize the number of license creations)
- `regularization`:  either `:ridge` or `:lasso`
"""
function sp_reformulation_path_jump(X, c, sp_graph::shortest_path_graph;
    sp_oracle = missing, reform_alg_parms::reformulation_path_parms = reformulation_path_parms())

    # Unpack parms
    @unpack lambda_max,
        lambda_min_ratio,
        num_lambda,
        solver,
        gurobiEnv,
        regularization,
        regularize_first_column_B,
        upper_bound_B_present,
        upper_bound_B,
        verbose,
        algorithm_type = reform_alg_parms

    @unpack sources, destinations, start_node, end_node, acyclic = sp_graph

    # Dimension check
    (p, n) = size(X)
    (d, n2) = size(c)
    if n != n2
        error("Dimensions of the input are mismatched.")
    end

    # Process graph
    nodes = unique(union(sources, destinations))
    n_nodes = length(nodes)
    n_edges = length(sources)
    if n_edges != d
        error("Dimensions of the input are mismatched.")
    end

    # Hard code sparse incidence matrix!
    I_vec = [sources; destinations]
    J_vec = [collect(1:n_edges); collect(1:n_edges)]
    V_vec = [-ones(n_edges); ones(n_edges)]
    A_mat = sparse(I_vec, J_vec, V_vec)
    A_mat_trans = transpose(A_mat)

    # Set up RHS
    b_vec = sparse(zeros(n_nodes))
    b_vec[start_node] = -1
    b_vec[end_node] = 1

    # Remove sparse
    A_mat = round.(Int, Matrix(A_mat))
    b_vec = round.(Int, Vector(b_vec))


    # Get oracle and w^*
    if ismissing(sp_oracle)
        sp_oracle = sp_flow_jump_setup(sources, destinations, start_node, end_node; solver = solver)
    end
    (z_star_data, w_star_data) = oracle_dataset(c, sp_oracle)


    # Start creating dual JuMP model
    if solver == :Gurobi
        mod = Model(with_optimizer(Gurobi.Optimizer, gurobiEnv))
    elseif solver == :Clp
        mod = Model(with_optimizer(Clp.Optimizer))
    else
        error("Not a valid solver: either :Clp or :Gurobi")
    end

    # Add variables
    @variable(mod, p_var[1:n_nodes, 1:n])
    if !acyclic
        @variable(mod, s_var[1:d, 1:n] >= 0)
    end
    if upper_bound_B_present
        @variable(mod, -upper_bound_B <= B_var[1:d, 1:p] <= upper_bound_B)
    else
        @variable(mod, B_var[1:d, 1:p])
    end

    # Add constraints
    for i = 1:n
        if acyclic
            @constraint(mod, -A_mat_trans*p_var[:,i] .>= c[:,i] - 2*B_var*X[:,i])
        else
            @constraint(mod, s_var[:,i] - A_mat_trans*p_var[:,i] .>= c[:,i] - 2*B_var*X[:,i])
        end
    end

    # Add objective part without regularization
    if regularization == :ridge
        obj_expr_noreg = QuadExpr()
    elseif regularization == :lasso
        obj_expr_noreg = AffExpr()
    else
        error("enter valid regularization: :ridge or :lasso")
    end
    for i = 1:n
        if acyclic
            cur_expr = -dot(b_vec, p_var[:,i]) + 2*(w_star_data[:,i]')*B_var*X[:,i] - z_star_data[i]
        else
            cur_expr = -dot(b_vec, p_var[:,i]) + dot(ones(d), s_var[:,i]) + 2*(w_star_data[:,i]')*B_var*X[:,i] - z_star_data[i]
        end
        add_to_expression!(obj_expr_noreg, cur_expr)
        #obj_expr_noreg = obj_expr_noreg + cur_expr
    end


    ### Add regularization part and solve for path
    if ismissing(lambda_max)
        lambda_max = (d/n)*(norm(X)^2)
    end

    # Construct lambda sequence
    if num_lambda == 1 && lambda_max == 0
        lambdas = [0.0]
    else
        lambda_min = lambda_max*lambda_min_ratio
        #println("Lambda min is $lambda_min")
        log_lambdas = collect(range(log(lambda_min), stop = log(lambda_max), length = num_lambda))
        lambdas = exp.(log_lambdas)
    end

    # Add theta variables for Lasso
    if regularization == :lasso
        @variable(mod, theta_var[1:d, 1:p])
        @constraint(mod, theta_var .>= B_var)
        @constraint(mod, theta_var .>= -B_var)
    end

    # Make list of B models and solve path
    B_soln_list = Matrix{Float64}[]
    for i = 1:num_lambda
        cur_lambda = lambdas[i]
        if verbose
            println("Trying lambda = $cur_lambda")
        end

        if regularization == :ridge && regularize_first_column_B
            obj_expr_full = obj_expr_noreg + n*(cur_lambda/2)*dot(B_var, B_var)
        elseif regularization == :ridge && !regularize_first_column_B
            obj_expr_full = obj_expr_noreg + n*(cur_lambda/2)*dot(B_var[:, 2:p], B_var[:, 2:p])
        elseif regularization == :lasso && regularize_first_column_B
            obj_expr_full = obj_expr_noreg + n*cur_lambda*sum(theta_var)
        elseif regularization == :lasso && !regularize_first_column_B
            obj_expr_full = obj_expr_noreg + n*cur_lambda*sum(theta_var[:, 2:p])
        else
            error("enter valid regularization: :ridge or :lasso")
        end
        @objective(mod, Min, obj_expr_full)

        # solve
        optimize!(mod)
        #z_ast = objective_value(mod)
        mod_status = termination_status(mod)
        B_ast = value.(B_var)

        if any(isnan, B_ast)
            # set B_ast to zero
            B_ast = zeros(d, p)

            # print reason
            println("We got NaNs in B_ast and the reason is:  $mod_status. The algoritim is $algorithm_type.")
        elseif Int(mod_status) != 1
            println("There are no NaNs in B_ast, but the model status is:  $mod_status. The algoritim is $algorithm_type.")
        end

        # if Int(mod_status) != 1
        #     println("Status of the model is: $mod_status")
        #     println("Current lambda is:  $cur_lambda")
        #
        #     B_val = value.(B_var)
        #     p_val = value.(p_var)
        #     s_val = value.(s_var)
        #     theta_val = value.(theta_var)
        #     println("Value of B_var is:  $B_val")
        #     println("Value of p_var is:  $p_val")
        #     println("Value of s_var is:  $s_val")
        #     println("Value of theta_var is:  $theta_val")
        # end

        push!(B_soln_list, B_ast)
    end

    return B_soln_list, lambdas
end


"""
    leastSquares_path_jump(X, c;
        reform_alg_parms = reformulation_path_parms())

Solves the empirical risk least squares problem using JuMP.
Solves over a grid of `num_lamba` regularization parameter values that are evenly spaced
on a log scale between `lambda_min_ratio*lambda_max` and `lambda_max`.
Both Ridge and Lasso are implemented. Returns `B_soln_list, lambdas` where `B_soln_list` is a list of
lienar models such that `B_soln_list[i]` is the solution with parameter `lambdas[i]`.

The default values for the algorithm parameters `reform_alg_parms` are (explanations are given below):
    lambda_max::Number = -1 (actual default is `(d/n)*(norm(X)^2)`)
    lambda_min_ratio::Float64 = 0.0001
    num_lambda::Int = 100
    solver::Symbol = :Gurobi
    regularization::Symbol = :ridge
    po_loss_function::Symbol = :leastSquares
    huber_delta::Float64 = 0.0

# Arguments
- `X`: p x n training set feature matrix
- `c`: d x n training set matrix of cost vectors
- `lambda_max`:  largest value of lambda, default is `(d/n)*(norm(X)^2)`
- `lambda_min_ratio`:  lambda_min is `lambda_max*lambda_min_ratio`
- `num_lambda`:  number of lambda values, evenly spaced on log scale between [lambda_min, lambda_max]
- `solver`:  either `:Gurobi` or `:Clp`
- `gurobiEnv` is a Gurobi enviornment which can be passed in (helpful to minimize the number of license creations)
- `regularization`:  either `:ridge` or `:lasso`
- `regularize_first_column_B`:  either true or false
- `po_loss_function`:  either `:leastSquares` or `:huber` or `:absolute`
- `huber_delta`:  the delta parameter for Huber loss
"""
function leastSquares_path_jump(X, c;
    reform_alg_parms::reformulation_path_parms = reformulation_path_parms())

    # Unpack parms
    @unpack lambda_max,
        lambda_min_ratio,
        num_lambda,
        solver,
        gurobiEnv,
        regularization,
        regularize_first_column_B,
        upper_bound_B,
        upper_bound_B_present,
        po_loss_function,
        huber_delta,
        verbose,
        algorithm_type = reform_alg_parms

    # Dimension check
    (p, n) = size(X)
    (d, n2) = size(c)
    if n != n2
        error("Dimensions of the input are mismatched.")
    end

    # Start creating JuMP model
    if solver == :Gurobi
        mod = Model(with_optimizer(Gurobi.Optimizer, gurobiEnv))
    elseif solver == :Clp
        mod = Model(with_optimizer(Clp.Optimizer))
    else
        error("Not a valid solver: either :Clp or :Gurobi")
    end

    # Add variables
    if upper_bound_B_present
        @variable(mod, -upper_bound_B <= B_var[1:d, 1:p] <= upper_bound_B)
    else
        @variable(mod, B_var[1:d, 1:p])
    end

    # Add least squares or Huber objective
    if po_loss_function == :leastSquares
        obj_expr_noreg = dot(c - B_var*X, c - B_var*X)
    elseif po_loss_function == :huber
        @variable(mod, w_var[1:d, 1:n] >= 0)
        @variable(mod, v_var[1:d, 1:n])
        @constraint(mod, c - B_var*X .<= v_var + w_var)
        @constraint(mod, -(c - B_var*X) .<= v_var + w_var)
        @constraint(mod, w_var .<= huber_delta*ones(d, n))

        obj_expr_noreg = dot(w_var, w_var) + 2*huber_delta*dot(v_var, ones(d, n))
    elseif po_loss_function == :absolute
        @variable(mod, w_var[1:d, 1:n] >= 0)
        @constraint(mod, c - B_var*X .<= w_var)
        @constraint(mod, -(c - B_var*X) .<= w_var)

        obj_expr_noreg = 2*dot(w_var, ones(d, n))
    else
        error("Enter a valid loss function: either :leastSquares or :huber or :absolute")
    end


    ### Add regularization part and solve for path
    if ismissing(lambda_max)
        lambda_max = (d/n)*(norm(X)^2)
    end

    # Construct lambda sequence
    if num_lambda == 1 && lambda_max == 0
        lambdas = [0.0]
    else
        lambda_min = lambda_max*lambda_min_ratio
        #println("Lambda min is $lambda_min")
        log_lambdas = collect(range(log(lambda_min), stop = log(lambda_max), length = num_lambda))
        lambdas = exp.(log_lambdas)
    end

    # Add theta variables for Lasso
    if regularization == :lasso
        @variable(mod, theta_var[1:d, 1:p])
        @constraint(mod, theta_var .>= B_var)
        @constraint(mod, theta_var .>= -B_var)
    end

    # Make list of B models and solve path
    B_soln_list = Matrix{Float64}[]
    for i = 1:num_lambda
        cur_lambda = lambdas[i]
        if verbose
            println("Trying lambda = $cur_lambda")
        end

        if regularization == :ridge && regularize_first_column_B
            obj_expr_full = obj_expr_noreg + n*cur_lambda*dot(B_var, B_var)
        elseif regularization == :ridge && !regularize_first_column_B
            obj_expr_full = obj_expr_noreg + n*cur_lambda*dot(B_var[:, 2:p], B_var[:, 2:p])
        elseif regularization == :lasso && regularize_first_column_B
            obj_expr_full = obj_expr_noreg + 2*n*cur_lambda*sum(theta_var)
        elseif regularization == :lasso && !regularize_first_column_B
            obj_expr_full = obj_expr_noreg + 2*n*cur_lambda*sum(theta_var[:, 2:p])
        else
            error("enter valid regularization: :ridge or :lasso")
        end
        @objective(mod, Min, obj_expr_full)

        # solve
        optimize!(mod)
        #z_ast = objective_value(mod)
        mod_status = termination_status(mod)
        B_ast = value.(B_var)

        if any(isnan, B_ast)
            # set B_ast to zero
            B_ast = zeros(d, p)

            # print reason
            println("We got NaNs in B_ast and the reason is:  $mod_status. The algoritim is $algorithm_type.")
        elseif Int(mod_status) != 1
            println("There are no NaNs in B_ast, but the model status is:  $mod_status. The algoritim is $algorithm_type.")
        end

        push!(B_soln_list, B_ast)
    end

    return B_soln_list, lambdas
end
