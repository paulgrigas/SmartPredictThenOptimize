#=
Cross validation tuning methods.
Dependencies: sgd.jl, reformulation.jl, spo_util.jl
=#

using Parameters, Gadfly

## Type for storing validation set algorithm parameters
@with_kw struct val_parms
    algorithm_type::Symbol = :spo_plus_sgd
    validation_set_percent::Float64 = 0.2
    validation_loss::Symbol = :spo_loss
    plot_results::Bool = false
    resolve_sgd::Bool = false
    resolve_sgd_accuracy::Float64 = 0.00001
    resolve_iteration_limit::Int = 50000
end

"""
    validation_set_alg(X::Matrix{Float64}, c::Matrix{Float64}, oracle;
        sp_graph = missing, train_ind = missing, validation_ind = missing,
        val_alg_parms::val_parms = val_parms(), path_alg_parms = sgd_path_parms())

Applies the validation set approach in order to tune the regularization parameter
for either least squares or SPO+ approach. Works with either the SGD path algorithm
for SPO+ or LS, or the shoretst path reformulation approach path algorithm. The user is allowed
to prespecify the training and validation sets by setting the optional parameters `train_ind` and `validation_ind`, which
should be mutually exclusvie and exhaustive subsets of `collect(1:n)`. If these are not provided, then
the algorithm will randomly use `validation_set_percent` of the `n` data points as the validation set.

The default values for the algorithm parameters `val_alg_parms` are (explanations are given below):
    algorithm_type::Symbol = :spo_plus_sgd
    validation_set_percent::Float64 = 0.2
    plot_results::Bool = false

Note that the overall default behavior is to run SGD to compute the SPO+ path.

# Arguments
- `X`: p x n training set feature matrix
- `c`: d x n training set matrix of cost vectors
- `oracle` is a function representing the linear optimization oracle for S, which always needs to be provided (even if doing reformulation)
- `sp_graph` is needed in the reformulation approach and is of the type `shortest_path_graph`. It provides information for reformulation approach (sources, destination, start_node, end_node)
- `train_ind` and `validation_ind` allow the user to specify the training and validation sets
- `algorithm_type` specifies which method to use and is one of:
    - `:spo_plus_sgd` = SGD to compute the path for SPO+ loss with ridge penalty
    - `:ls_sgd` = SGD to compute the path for least squares loss with ridge penalty
    - `:sp_spo_plus_reform` = reformulation approach for SPO+ loss on shortest path problem
    - `:ls_jump` = use JuMP and a solver to compute the least squares path
- `validation_set_percent` = percent of data to use for validations set with indicies are not provided
- If `plot_results` are true, then generate plots and save to file
- `path_alg_parms` specify the parameters of the path solver (either SGD or reformulation/JuMP) and should be of the type `sgd_path_parms` or `reformulation_path_parms`
"""
function validation_set_alg(X::Matrix{Float64}, c::Matrix{Float64}, oracle;
    sp_graph = missing, train_ind = missing, validation_ind = missing,
    val_alg_parms::val_parms = val_parms(), path_alg_parms = sgd_path_parms())

    # Get validation set parms
    @unpack algorithm_type,
        validation_set_percent,
        plot_results,
        validation_loss,
        resolve_sgd,
        resolve_sgd_accuracy,
        resolve_iteration_limit = val_alg_parms

    # Check for validitiy of path_alg_parms
    if typeof(path_alg_parms) != sgd_path_parms && typeof(path_alg_parms) != reformulation_path_parms
        error("path_alg_parms is not valid, should be either sgd_path_parms or reformulation_path_parms")
    end

    # Check for sp_graph
    if !ismissing(sp_graph)
        @unpack sources, destinations, start_node, end_node = sp_graph
    end

    # Get dimensions of input
    (p, n) = size(X)
    (d, n2) = size(c)

    # split into train/validation if not given already
    if ismissing(train_ind) && ismissing(validation_ind)
        validation_ind = randsubseq(collect(1:n), validation_set_percent)
        train_ind = setdiff(collect(1:n), validation_ind)
    end

    X_train = X[:, train_ind]
    X_validation = X[:, validation_ind]
    c_train = c[:, train_ind]
    c_validation = c[:, validation_ind]

    # train models
    if algorithm_type == :spo_plus_sgd
        B_soln_list, lambdas = spoPlus_sgd_path(X_train, c_train, oracle; path_alg_parms = path_alg_parms)
    elseif algorithm_type == :ls_sgd
        B_soln_list, lambdas = leastSquares_sgd_path(X_train, c_train, oracle; path_alg_parms = path_alg_parms)
    elseif algorithm_type == :sp_spo_plus_reform
        B_soln_list, lambdas = sp_reformulation_path_jump(X_train, c_train, sp_graph; sp_oracle = oracle, reform_alg_parms = path_alg_parms)
    elseif algorithm_type == :ls_jump
        B_soln_list, lambdas = leastSquares_path_jump(X_train, c_train; reform_alg_parms = path_alg_parms)
    else
        error("Enter a valid algorithm type")
    end

    # get best model
    num_lambda = length(lambdas)
    validation_loss_list = zeros(num_lambda)
    for i = 1:num_lambda
        if validation_loss == :spo_loss
            validation_loss_list[i] = spo_loss(B_soln_list[i], X_validation, c_validation, oracle)
        elseif validation_loss == :spo_plus_loss
            validation_loss_list[i] = spo_plus_loss(B_soln_list[i], X_validation, c_validation, oracle)
        elseif validation_loss == :least_squares_loss
            validation_loss_list[i] = least_squares_loss(B_soln_list[i], X_validation, c_validation)
        elseif validation_loss == :absolute_loss
            validation_loss_list[i] = absolute_loss(B_soln_list[i], X_validation, c_validation)
        elseif validation_loss == :huber_loss
            validation_loss_list[i] = huber_loss(B_soln_list[i], X_validation, c_validation, path_alg_parms.huber_delta)
        elseif validation_loss == :hamming_loss
            (z_validation, w_validation) = oracle_dataset(c_validation, oracle)
            (d_wval, n_wval) = size(w_validation)
            c_ham_validation = ones(d_wval, n_wval) - w_validation
            validation_loss_list[i] = spo_loss(B_soln_list[i], X_validation, c_ham_validation, oracle)
        else
            error("Enter a valid validation set loss function.")
        end
    end

    if plot_results
        val_plot = plot(x = log.(lambdas), y = validation_loss_list, Geom.line,
        Guide.xlabel("Log(Lambda)"), Guide.ylabel("Validation Set SPO Loss"))
        img = SVG("plots/validation_set_plot_spo_loss.svg", 6inch, 4inch)
        draw(img, val_plot)

        # Plot SPO+ or least squares loss as well
        surrogate_loss = zeros(num_lambda)
        if algorithm_type == :spo_plus_sgd || algorithm_type == :sp_spo_plus_reform
            for i = 1:num_lambda
                surrogate_loss[i] = spo_plus_loss(B_soln_list[i], X_validation, c_validation, oracle)
            end
            val_plot2 = plot(x = log.(lambdas), y = surrogate_loss, Geom.line,
            Guide.xlabel("Log(Lambda)"), Guide.ylabel("Validation Set SPO+ Loss"))
            img = SVG("plots/validation_set_plot_spoplus_loss.svg", 6inch, 4inch)
            draw(img, val_plot2)
        elseif algorithm_type == :ls_sgd || algorithm_type == :ls_jump
            for i = 1:num_lambda
                surrogate_loss[i] = least_squares_loss(B_soln_list[i], X_validation, c_validation)
            end
            val_plot2 = plot(x = log.(lambdas), y = surrogate_loss, Geom.line,
            Guide.xlabel("Log(Lambda)"), Guide.ylabel("Validation Set Least Squares Loss"))
            img = SVG("plots/validation_set_plot_leastsquares_loss.svg", 6inch, 4inch)
            draw(img, val_plot2)
        end
    end

    best_ind = argmin(validation_loss_list)
    best_lambda = lambdas[best_ind]
    best_B_matrix = B_soln_list[best_ind]

    if resolve_sgd && algorithm_type == :spo_plus_sgd
        # Make SGD Alg parms
        n_train = length(train_ind)
        second_moment_bound = (1/n_train)*(norm(X_train)^2)
        num_iters = ceil(Int, (2*second_moment_bound)/(best_lambda*resolve_sgd_accuracy)) + 1
        num_iters = min(num_iters, resolve_iteration_limit)
        sgd_alg_parms = sgd_parms(grad_type = path_alg_parms.grad_type, lambda = best_lambda,
            numiter = num_iters, batchsize = path_alg_parms.batchsize, step_type = :short)

        B_spo_plus_final, spo_sgd_history_trace_final = spoPlus_sgd(X_train, c_train, oracle;
            alg_parms = sgd_alg_parms, B_init = best_B_matrix)
    else
        B_spo_plus_final = best_B_matrix
    end

    return B_spo_plus_final, best_lambda
end
