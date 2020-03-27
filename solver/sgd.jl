#=
SGD algorithm for both SPO plus and least squares optimization problems.
Dependencies:  spo_util.jl
=#

using Parameters

## Type for storing SGD history
struct SGDHistory
    spo_train_history
    spo_plus_train_history
    spo_holdout_history
    returned_iter
end

## Type for SGD algorithm parameters with default values
@with_kw struct sgd_parms
    grad_type::Symbol = :stochastic
    lambda::Float64 = 0
    numiter::Int = 1000
    batchsize::Int = 10
    step_type::Symbol = :long_dynamic
    long_factor::Float64 = 0.1
    holdout_set::Bool = false
    holdout_period::Int = 1
    history::Bool = false
    history_period::Int = 1
end

"""
    spoPlus_sgd(X, c, oracle; alg_parms = sgd_parms(),
        X_holdout = [], c_holdout = [], B_init = zeros(d, p))

Solve the SPO+ problem with stochastic gradient descent. This is problem (18) from the paper.
Return the B matrix and an object of type SGDHistory. As described in the paper, the method always
returns an averaged iterate. If `holdout_set` is true, then a holdout set is used and the
method returns the averaged iterate with the best holdout set performance evaluated at every
mutiple of `holdout_period`. Otherwise, the method returns the final averaged iterate. If
`history` is true then the SPO loss training set, SPO+ loss training set, and SPO loss
holdout set values are also recorded at each mutiple of `history_period` and are returned as well.

The default values for the algorithm parameters `alg_parms` are (explanations are given below):
    grad_type::Symbol = :stochastic
    lambda::Float64 = 0
    numiter::Int = 1000
    batchsize::Int = 10
    step_type::Symbol = :long_dynamic
    long_factor::Float64 = 0.1
    holdout_set::Bool = false
    holdout_period::Int = 1
    history::Bool = false
    history_period::Int = 1

# Arguments
- `X`: p x n training set feature matrix
- `c`: d x n training set matrix of cost vectors
- `oracle` is the linear optimization oracle for S
- `lambda` is the regularization parameter for the ridge penalty (only ridge implemented)
- `numiter` is the number of iterations of the method. Note that the method runs for t = 0, 1, 2, ... numiter-1 iterations and returns `\\bar{B}_{numiter - 1}`
- `step_type` determines the type of step-size and is one of:
    - `short_practical` = 2/((`iter` + 2))
    - `:short` = 2/(`lambda` * (`iter` + 2)) from Lacoste-Julien et al. paper
    - `:long_static` = `long_factor`/sqrt(`numiter` + 1) from Nemirovski et al. paper
    - `:long_dynamic` = `long_factor`/sqrt(`iter` + 1) from Nemirovski et al. paper
    - `:long_static_normalized` = same as `:long_static` but also divide by norm of subgradient
    - `:long_dynamic_normalized` = same as `:long_dynamic` but also divide by norm of subgradient
- `grad_type` determines the type of subgradient computation and is one of:
    - `:deterministic` = deterministic SPO+ subgradients
    - `:stochastic` = stochastic SPO+ subgradients using `batchsize` samples at each iteration
    - `:deterministic_LS` = same as `deterministic` but replace SPO+ loss with least-squares loss
    - `:stochastic_LS` = same as `stochastic` but replace SPO+ loss with least-squares loss
- `X_holdout` and `c_holdout` are the holdout set data
- `B_init` is the initial iterate which defaults to all zeros
"""
function spoPlus_sgd(X::Matrix{Float64}, c::Matrix{Float64}, oracle;
    alg_parms::sgd_parms = sgd_parms(),
    X_holdout=[], c_holdout=[], B_init = zeros(d, p))

    # First unpack the parameters
    @unpack grad_type,
        lambda,
        numiter,
        batchsize,
        step_type,
        long_factor,
        holdout_set,
        holdout_period,
        history,
        history_period = alg_parms

    # Dimension check
    (p, n) = size(X)
    (d, n2) = size(c)
    if n != n2
        error("Dimensions of the input are mismatched.")
    end

    # pre-process to get z^\ast(c_i) and w^\ast(c_i)
    (z_star_data, w_star_data) = oracle_dataset(c, oracle)

    #=
    Code for determining the subgradient and step-size functions, which
    depends on specific input parameters.
    =#

    # Generate function for computing subgradient based on grad_type
    # B_new is the current iterate
    if grad_type == :deterministic
        function subgrad_deterministic(B_new)
            G_new = zeros(d, p)
            for i = 1:n
                spoplus_cost_vec = 2*B_new*X[:,i] - c[:,i]
                (z_oracle, w_oracle) = oracle(spoplus_cost_vec)
                w_star_diff = w_star_data[:,i] - w_oracle
                G_new = G_new + 2*w_star_diff*(X[:,i]')
            end
            G_new = (1/n)*G_new + lambda*B_new
            return G_new
        end
        subgrad = subgrad_deterministic
    elseif grad_type == :stochastic
        function subgrad_stochastic(B_new)
            G_new = zeros(d, p)
            for j = 1:batchsize
                i = rand(1:n)
                spoplus_cost_vec = 2*B_new*X[:,i] - c[:,i]
                (z_oracle, w_oracle) = oracle(spoplus_cost_vec)
                w_star_diff = w_star_data[:,i] - w_oracle
                G_new = G_new + 2*w_star_diff*(X[:,i]')
            end
            G_new = (1/batchsize)*G_new + lambda*B_new
            return G_new
        end
        subgrad = subgrad_stochastic
    elseif grad_type == :deterministic_LS
        function subgrad_deterministic_LS(B_new)
            G_new = zeros(d, p)
            for i = 1:n
                residuals = B_new*X[:,i] - c[:,i]
                G_new = G_new + residuals*(X[:,i]')
            end
            G_new = (1/n)*G_new + lambda*B_new
            return G_new
        end
        subgrad = subgrad_deterministic_LS
    elseif grad_type == :stochastic_LS
        function subgrad_stochastic_LS(B_new)
            G_new = zeros(d, p)
            for j = 1:batchsize
                i = rand(1:n)
                residuals = B_new*X[:,i] - c[:,i]
                G_new = G_new + residuals*(X[:,i]')
            end
            G_new = (1/batchsize)*G_new + lambda*B_new
            return G_new
        end
        subgrad = subgrad_stochastic_LS
    else
        error("Enter a valid grad_type.")
    end

    # Generate function for computing step-size
    # iter is the iteration counter that starts at 0 and ends at numiter
    # G_new is the current subgradient computed at iteration iter
    if step_type == :short
        function step_size_short(iter, G_new)
            return 2/(lambda*(iter + 2))
        end
        step_size = step_size_short
    elseif step_type == :short_practical
        function step_size_short_practical(iter, G_new)
            return 2/(iter + 2)
        end
        step_size = step_size_short_practical
    elseif step_type == :long_static
        function step_size_long_static(iter, G_new)
            return long_factor/sqrt(numiter + 1)
        end
        step_size = step_size_long_static
    elseif step_type == :long_dynamic
        function step_size_long_dynamic(iter, G_new)
            return long_factor/sqrt(iter + 1)
        end
        step_size = step_size_long_dynamic
    elseif step_type == :long_static_normalized
        function step_size_long_static_normalized(iter, G_new)
            G_norm = norm(G_new)
            return long_factor/(G_norm*sqrt(numiter + 1))
        end
        step_size = step_size_long_static_normalized
    elseif step_type == :long_dynamic_normalized
        function step_size_long_dynamic_normalized(iter, G_new)
            G_norm = norm(G_new)
            return long_factor/(G_norm*sqrt(iter + 1))
        end
        step_size = step_size_long_dynamic_normalized
    else
        error("Enter a valid step_type.")
    end


    #=
    Pre-processing for holdout set and history options
    =#

    if holdout_set
        (p_holdout, n_holdout) = size(X_holdout)
        (d_holdout, n_holdout_2) = size(c_holdout)
        if n_holdout != n_holdout_2 || p_holdout != p || d_holdout != d
            error("Dimensions of the input are mismatched.")
        end

        # pre-process to get w^\ast(c_i) and z^\ast(c_i) for the holdout set
        (z_star_holdout, w_star_holdout) = oracle_dataset(c_holdout, oracle)

        # function to compute the SPO loss on the holdout set
        holdout_set_spo(B_new) = spo_loss(B_new, X_holdout, c_holdout, oracle;
            z_star = z_star_holdout)

        # variables to track the best holdout set iterate
        B_cur_best_holdout = zeros(d, p)
        cur_best_spo_holdout = Inf
        spo_holdout_history = zeros(numiter) - 1
    end

    if history
        # function to compute the SPO loss on the training set
        training_set_spo(B_new) = spo_loss(B_new, X, c, oracle; z_star=z_star_data)

        # function to compute the SPO plus loss on the training set
        training_set_spo_plus(B_new) = spo_plus_loss(B_new, X, c, oracle;
            z_star = z_star_data, w_star = w_star_data)

        # variables to track history
        spo_train_history = zeros(numiter) - 1
        spo_plus_train_history = zeros(numiter) - 1
    end


    #=
    Begin logic of the subgradient method
    =#

    # initialize iterates
    B_iter = B_init
    B_avg_iter = B_init
    step_size_sum = 0
    returned_iter = numiter - 1 # default if we do not do holdout

    for iter = 0:(numiter - 1)
        # call subgradient and step-size functions
        G_iter = subgrad(B_iter)
        step_iter = step_size(iter, G_iter)

        # Update average iterate and then current iterate
        # Note the lag, the average doesn't use the latest iterate :(
        step_size_sum = step_size_sum + step_iter
        step_avg = step_iter/step_size_sum
        B_avg_iter = (1 - step_avg)*B_avg_iter + step_avg*B_iter

        B_iter = B_iter - step_iter*G_iter


        # Update holdout set best and history
        if holdout_set && rem(iter, holdout_period) == 0
            spo_holdout_iter = holdout_set_spo(B_avg_iter)
            spo_holdout_history[iter + 1] = spo_holdout_iter
            if spo_holdout_iter < cur_best_spo_holdout
                cur_best_spo_holdout = spo_holdout_iter
                B_cur_best_holdout = B_avg_iter
                returned_iter = iter
            end
        end

        if history && rem(iter, history_period) == 0
            spo_train_history[iter + 1] = training_set_spo(B_avg_iter)
            spo_plus_train_history[iter + 1] = training_set_spo_plus(B_avg_iter)
        end
    end

    ## Generate history object to return
    if holdout_set && history
        sgd_history_trace = SGDHistory(spo_train_history, spo_plus_train_history, spo_holdout_history, returned_iter)
    elseif !holdout_set && history
        sgd_history_trace = SGDHistory(spo_train_history, spo_plus_train_history, [], returned_iter)
    else
        sgd_history_trace = SGDHistory([], [], [], returned_iter)
    end

    ## return B and history
    if holdout_set
        return B_cur_best_holdout, sgd_history_trace
    else
        return B_avg_iter, sgd_history_trace
    end
end


"""
    leastSquares_sgd(X, c, oracle; alg_parms = sgd_parms(),
        X_holdout=[], c_holdout=[], B_init = zeros(d, p))

Wrapper around `spoPlus_sgd` for least-squares regression. Simply defaults `grad_type`
to `stochastic_LS`.
"""
function leastSquares_sgd(X::Matrix{Float64}, c::Matrix{Float64}, oracle;
    alg_parms::sgd_parms = sgd_parms(grad_type = :stochastic_LS),
    X_holdout=[], c_holdout=[], B_init = zeros(d, p))

    # overwrite grad_type just in case
    alg_parms_new = sgd_parms(alg_parms; grad_type = :stochastic_LS)

    return spoPlus_sgd(X, c, oracle; alg_parms = alg_parms_new,
        X_holdout = X_holdout, c_holdout = c_holdout, B_init = B_init)
end


# SGD Path parameters (batchsize and grad_type are the only sgd_parms that can be customized for the path algorithm)
@with_kw struct sgd_path_parms
    lambda_max::Union{Symbol, Float64} = :practical
    lambda_min_ratio::Float64 = 0.0001
    num_lambda::Int = 100
    grad_type::Symbol = :stochastic
    second_moment_bound::Symbol = :practical
    obj_accuracy::Float64 = 0.0001
    batchsize::Int = 10
    iteration_limit::Int = 10000
    step_type::Symbol = :short
    verbose::Bool = false
end

"""
    spoPlus_sgd_path(X, c, oracle;
        path_alg_parms::sgd_path_parms = sgd_path_parms())

Solves the ridge regularization path using SGD over a grid of `num_lamba`
regularization parameter values that are evenly spaced on a log scale between
`lambda_min_ratio*lambda_max` and `lambda_max`. Returns `B_soln_list, lambdas`
where `B_soln_list` is a list of linear models such that `B_soln_list[i]` is the
solution with parameter `lambdas[i]`.

The default values for the algorithm parameters `path_alg_parms` are (explanations are given below):
    lambda_max::Symbol = :practical
    lambda_min_ratio::Float64 = 0.0001
    num_lambda::Int = 100
    grad_type::Symbol = :stochastic
    second_moment_bound::Symbol = :practical
    obj_accuracy::Float64 = 0.0001
    batchsize::Int = 10

# Arguments
- `X`: p x n training set feature matrix
- `c`: d x n training set matrix of cost vectors
- `oracle` is the linear optimization oracle for S
- `lambda_max`:  largest value of lambda, default is `:practical` which yields `lambda_max = (d/n)*(norm(X)^2)`
- `lambda_min_ratio`:  lambda_min is `lambda_max*lambda_min_ratio`
- `num_lambda`:  number of lambda values, evenly spaced on log scale between [lambda_min, lambda_max]
- `grad_type` determines the type of subgradient computation and is one of:
    - `:deterministic` = deterministic SPO+ subgradients
    - `:stochastic` = stochastic SPO+ subgradients using `batchsize` samples at each iteration
    - `:deterministic_LS` = same as `deterministic` but replace SPO+ loss with least-squares loss
    - `:stochastic_LS` = same as `stochastic` but replace SPO+ loss with least-squares loss
- `second_moment_bound` is either `:theory` as prescribed by Lacoste-Julien et al. paper or `:practical` which is `(1/n)*(norm(X)^2)`
- `obj_accuracy` is the desired objective function accuracy along the path
- `batchsize` is the number of samples per iteration in SGD
"""
function spoPlus_sgd_path(X::Matrix{Float64}, c::Matrix{Float64}, oracle;
    path_alg_parms::sgd_path_parms = sgd_path_parms())

    # Unpack parameters
    @unpack lambda_max,
        lambda_min_ratio,
        num_lambda,
        grad_type,
        second_moment_bound,
        obj_accuracy,
        batchsize,
        iteration_limit,
        step_type,
        verbose = path_alg_parms

    # Get dimensions of input
    (p, n) = size(X)
    (d, n2) = size(c)

    # assumes diam(S) \leq sqrt(d)
    if second_moment_bound == :practical
        second_moment_bound = (1/n)*(norm(X)^2)
    elseif second_moment_bound == :theory
        second_moment_bound = 16*(d/n)*(norm(X)^2)
    else
        error("Enter a valid second_moment_bound:  either :practical or :theory")
    end

    if lambda_max == :practical
        lambda_max = (d/n)*(norm(X)^2)
    elseif lambda_max == :theory
        lambda_max = (2*second_moment_bound)/obj_accuracy
    elseif typeof(lambda_max) != Float64 || lambda_max <= 0
        error("Enter a valid lambda_max:  either :practical or :theory")
    end

    # Construct lambda sequence
    lambda_min = lambda_max*lambda_min_ratio
    log_lambdas = collect(range(log(lambda_min), stop = log(lambda_max), length = num_lambda))
    lambdas = exp.(log_lambdas)

    # Make list of B models
    B_soln_list = Matrix{Float64}[]
    # Run SGD on path
    B_spo_plus = zeros(d, p)
    for i = 1:num_lambda
        cur_lambda = lambdas[i]
        if verbose
            println("Trying lambda = $cur_lambda")
        end
        num_iters = ceil(Int, (2*second_moment_bound)/(cur_lambda*obj_accuracy)) + 1
        num_iters = min(num_iters, iteration_limit)

        # Make SGD Alg parms
        sgd_alg_parms = sgd_parms(grad_type = grad_type, lambda = cur_lambda,
            numiter = num_iters, batchsize = batchsize, step_type = step_type)

        B_spo_plus, spo_sgd_history_trace = spoPlus_sgd(X, c, oracle;
            alg_parms = sgd_alg_parms, B_init = B_spo_plus)

        push!(B_soln_list, B_spo_plus)
    end

    return B_soln_list, lambdas
end

function leastSquares_sgd_path(X::Matrix{Float64}, c::Matrix{Float64}, oracle;
    path_alg_parms::sgd_path_parms = sgd_path_parms(grad_type = :stochastic_LS))

    # overwrite grad_type just in case
    path_alg_parms_new = sgd_path_parms(path_alg_parms; grad_type = :stochastic_LS)

    return spoPlus_sgd_path(X, c, oracle; path_alg_parms = path_alg_parms_new)
end
