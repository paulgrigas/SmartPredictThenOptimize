#=
Functions for generating a single replication of a particular experiment
Dependencies: util.jl, sgd.jl, reformulation.jl, random_forests_po.jl, validation_set.jl
=#

using Parameters, DataFrames, Statistics, Random, Gurobi

## Type for strong the results of an experiment replication
@with_kw mutable struct replication_results
    SPOplus_spoloss_test::Union{Float64, Missing} = missing # SPO loss of SPO+ SGD on test set
    LS_spoloss_test::Union{Float64, Missing} = missing # SPO loss of LS SGD on test set
    SSVM_spoloss_test::Union{Float64, Missing} = missing # SPO loss of SSVM-Hamming SGD on test set
    RF_spoloss_test::Union{Float64, Missing} = missing # SPO loss of random forests on test set
    Absolute_spoloss_test::Union{Float64, Missing} = missing # SPO loss of ell_1 on test set
    Huber_spoloss_test::Union{Float64, Missing} = missing  # SPO loss of Huber on test set
    Baseline_spoloss_test::Union{Float64, Missing} = missing # SPO loss of baseline that returns mean of training data, on test set
    SPOplus_hammingloss_test::Union{Float64, Missing} = missing # Hamming loss of SPO+ SGD on test set
    LS_hammingloss_test::Union{Float64, Missing} = missing # Hamming loss of LS SGD on test set
    SSVM_hammingloss_test::Union{Float64, Missing} = missing # Hamming loss of SSVM SGD on test set
    RF_hammingloss_test::Union{Float64, Missing} = missing # Hamming loss of RF on test set
    Absolute_hammingloss_test::Union{Float64, Missing} = missing # Hamming loss of ell_1 on test set
    Huber_hammingloss_test::Union{Float64, Missing} = missing  # Hamming loss of Huber on test set
    Baseline_hammingloss_test::Union{Float64, Missing} = missing # Hamming loss of baseline on test set
    zstar_avg_test::Union{Float64, Missing} = missing # Average of z^*(c_i) on testing set
end

"""
Make a blank data frame for storing the results of the experiment.
"""
function make_blank_complete_df()
    results_df = DataFrame(
        grid_dim = Int64[],
        n_train = Int64[],
        n_holdout = Int64[],
        n_test = Int64[],
        p_features = Int64[],
        polykernel_degree = Int64[],
        polykernel_noise_half_width = Float64[],
        SPOplus_spoloss_test = Union{Float64, Missing}[], # SPO loss of SPO+ SGD on test set
        LS_spoloss_test = Union{Float64, Missing}[],  # SPO loss of LS SGD on test set
        SSVM_spoloss_test = Union{Float64, Missing}[], # SPO loss of SSVM-Hamming SGD on test set
        RF_spoloss_test = Union{Float64, Missing}[], # SPO loss of random forests on test set
        Absolute_spoloss_test = Union{Float64, Missing}[], # SPO loss of ell_1 on test set
        Huber_spoloss_test = Union{Float64, Missing}[],  # SPO loss of Huber on test set
        Baseline_spoloss_test = Union{Float64, Missing}[], # SPO loss of baseline that returns mean of training data, on test set
        SPOplus_hammingloss_test = Union{Float64, Missing}[], # Hamming loss of SPO+ SGD on test set
        LS_hammingloss_test = Union{Float64, Missing}[], # Hamming loss of LS SGD on test set
        SSVM_hammingloss_test = Union{Float64, Missing}[], # Hamming loss of SSVM SGD on test set
        RF_hammingloss_test = Union{Float64, Missing}[], # Hamming loss of RF on test set
        Absolute_hammingloss_test = Union{Float64, Missing}[], # Hamming loss of ell_1 on test set
        Huber_hammingloss_test = Union{Float64, Missing}[],  # Hamming loss of Huber on test set
        Baseline_hammingloss_test = Union{Float64, Missing}[], # Hamming loss of baseline on test set
        zstar_avg_test = Union{Float64, Missing}[]) # Average of z^*(c_i) on testing set

    return results_df
end

"""
Make a new data frame that contains one row of results for an experiment instance that was just run.
"""
function build_complete_row(grid_dim,
    n_train, n_holdout, n_test,
    p_features, polykernel_degree, polykernel_noise_half_width, results_struct)

    df = make_blank_complete_df()

    push!(df, [
        grid_dim,
        n_train,
        n_holdout,
        n_test,
        p_features,
        polykernel_degree,
        polykernel_noise_half_width,
        results_struct.SPOplus_spoloss_test,
        results_struct.LS_spoloss_test,
        results_struct.SSVM_spoloss_test,
        results_struct.RF_spoloss_test,
        results_struct.Absolute_spoloss_test,
        results_struct.Huber_spoloss_test,
        results_struct.Baseline_spoloss_test,
        results_struct.SPOplus_hammingloss_test,
        results_struct.LS_hammingloss_test,
        results_struct.SSVM_hammingloss_test,
        results_struct.RF_hammingloss_test,
        results_struct.Absolute_hammingloss_test,
        results_struct.Huber_hammingloss_test,
        results_struct.Baseline_hammingloss_test,
        results_struct.zstar_avg_test])

    return df
end

"""
Run one instance of the shortest path experiment. Uses the reformulation approach.
`num_lambda` is the number of lambdas used on the grid for each method with regularization.
Returns a replication_results struct.
"""
function shortest_path_replication(grid_dim,
    n_train, n_holdout, n_test,
    p_features, polykernel_degree, polykernel_noise_half_width;
    num_lambda = 10, lambda_max = missing, lambda_min_ratio = 0.0001, regularization = :ridge,
    gurobiEnvOracle = Gurobi.Env(), gurobiEnvReform = Gurobi.Env(), different_validation_losses = false,
    include_rf = true)

    # Get oracle
    sources, destinations = convert_grid_to_list(grid_dim, grid_dim)
    d_feasibleregion = length(sources)
    sp_oracle = sp_flow_jump_setup(sources, destinations, 1, grid_dim^2; gurobiEnv = gurobiEnvOracle)
    sp_graph = shortest_path_graph(sources = sources, destinations = destinations,
        start_node = 1, end_node = grid_dim^2, acyclic = true)


    # Generate Data
    B_true = rand(Bernoulli(0.5), d_feasibleregion, p_features)

    (X_train, c_train) = generate_poly_kernel_data_simple(B_true, n_train, polykernel_degree, polykernel_noise_half_width)
    (X_validation, c_validation) = generate_poly_kernel_data_simple(B_true, n_holdout, polykernel_degree, polykernel_noise_half_width)
    (X_test, c_test) = generate_poly_kernel_data_simple(B_true, n_test, polykernel_degree, polykernel_noise_half_width)

    # Add intercept in the first row of X
    X_train = vcat(ones(1,n_train), X_train)
    X_validation = vcat(ones(1,n_holdout), X_validation)
    X_test = vcat(ones(1,n_test), X_test)

    # Get Hamming labels
    (z_train, w_train) = oracle_dataset(c_train, sp_oracle)
    (z_validation, w_validation) = oracle_dataset(c_validation, sp_oracle)
    (z_test, w_test) = oracle_dataset(c_test, sp_oracle)

    c_ham_train = ones(d_feasibleregion, n_train) - w_train
    c_ham_validation = ones(d_feasibleregion, n_holdout) - w_validation
    c_ham_test = ones(d_feasibleregion, n_test) - w_test

    # Put train + validation together
    X_both = hcat(X_train, X_validation)
    c_both = hcat(c_train, c_validation)
    c_ham_both = hcat(c_ham_train, c_ham_validation)
    train_ind = collect(1:n_train)
    validation_ind = collect((n_train + 1):(n_train + n_holdout))

    # Set validation losses
    if different_validation_losses
        spo_plus_val_loss = :spo_loss
        ls_val_loss = :least_squares_loss
        ssvm_val_loss = :hamming_loss
        absolute_val_loss = :absolute_loss
        huber_val_loss = :huber_loss
    else
        spo_plus_val_loss = :spo_loss
        ls_val_loss = :spo_loss
        ssvm_val_loss = :spo_loss
        absolute_val_loss = :spo_loss
        huber_val_loss = :spo_loss
    end

    ### Algorithms ###

    # SPO+
    best_B_SPOplus, best_lambda_SPOplus = validation_set_alg(X_both, c_both, sp_oracle; sp_graph = sp_graph,
        train_ind = train_ind, validation_ind = validation_ind,
        val_alg_parms = val_parms(algorithm_type = :sp_spo_plus_reform, validation_loss = spo_plus_val_loss),
        path_alg_parms = reformulation_path_parms(num_lambda = num_lambda, lambda_max = lambda_max, regularization = regularization,
            gurobiEnv = gurobiEnvReform, lambda_min_ratio = lambda_min_ratio, algorithm_type = :SPO_plus))

    # Least squares
    best_B_leastSquares, best_lambda_leastSquares = validation_set_alg(X_both, c_both, sp_oracle; sp_graph = sp_graph,
        train_ind = train_ind, validation_ind = validation_ind,
        val_alg_parms = val_parms(algorithm_type = :ls_jump, validation_loss = ls_val_loss),
        path_alg_parms = reformulation_path_parms(num_lambda = num_lambda, lambda_max = lambda_max, regularization = regularization,
            gurobiEnv = gurobiEnvReform, lambda_min_ratio = lambda_min_ratio, algorithm_type = :leastSquares))

    # SSVM Hamming
    best_B_SSVM, best_lambda_SSVM = validation_set_alg(X_both, c_ham_both, sp_oracle; sp_graph = sp_graph,
        train_ind = train_ind, validation_ind = validation_ind,
        val_alg_parms = val_parms(algorithm_type = :sp_spo_plus_reform, validation_loss = ssvm_val_loss),
        path_alg_parms = reformulation_path_parms(num_lambda = num_lambda, lambda_max = lambda_max, regularization = regularization,
            gurobiEnv = gurobiEnvReform, lambda_min_ratio = lambda_min_ratio, algorithm_type = :SSVM_hamming))

    # RF
    if include_rf
        rf_mods = train_random_forests_po(X_train, c_train;
            rf_alg_parms = rf_parms())
    end

    # Absolute
    best_B_Absolute, best_lambda_Absolute = validation_set_alg(X_both, c_both, sp_oracle; sp_graph = sp_graph,
        train_ind = train_ind, validation_ind = validation_ind,
        val_alg_parms = val_parms(algorithm_type = :ls_jump, validation_loss = absolute_val_loss),
        path_alg_parms = reformulation_path_parms(num_lambda = num_lambda, lambda_max = lambda_max, regularization = regularization,
            po_loss_function = :absolute, gurobiEnv = gurobiEnvReform, lambda_min_ratio = lambda_min_ratio, algorithm_type = :Absolute))

    # Huber
    best_B_fake, best_lambda_fake = validation_set_alg(X_both, c_both, sp_oracle; sp_graph = sp_graph,
        train_ind = train_ind, validation_ind = validation_ind,
        val_alg_parms = val_parms(algorithm_type = :ls_jump, validation_loss = ls_val_loss),
        path_alg_parms = reformulation_path_parms(lambda_max = 0.0001, regularization = regularization,
            num_lambda = 2, gurobiEnv = gurobiEnvReform, algorithm_type = :Huber_LS_fake))

    fake_ls_list = abs.(vec(c_train - best_B_fake*X_train))
    delta_from_fake_ls = median(fake_ls_list)

    best_B_Huber, best_lambda_Huber = validation_set_alg(X_both, c_both, sp_oracle; sp_graph = sp_graph,
        train_ind = train_ind, validation_ind = validation_ind,
        val_alg_parms = val_parms(algorithm_type = :ls_jump, validation_loss = huber_val_loss),
        path_alg_parms = reformulation_path_parms(num_lambda = num_lambda, lambda_max = lambda_max, regularization = regularization, po_loss_function = :huber,
            huber_delta = delta_from_fake_ls, lambda_min_ratio = lambda_min_ratio, gurobiEnv = gurobiEnvReform, algorithm_type = :Huber))


    # Baseline
    c_bar_train = mean(c_train, dims=2)


    ### Populate final results ###
    final_results = replication_results()

    final_results.SPOplus_spoloss_test = spo_loss(best_B_SPOplus, X_test, c_test, sp_oracle)
    final_results.LS_spoloss_test = spo_loss(best_B_leastSquares, X_test, c_test, sp_oracle)
    final_results.SSVM_spoloss_test = spo_loss(best_B_SSVM, X_test, c_test, sp_oracle)

    if include_rf
        rf_preds_test = predict_random_forests_po(rf_mods, X_test)
        final_results.RF_spoloss_test = spo_loss(Matrix(1.0I, d_feasibleregion, d_feasibleregion), rf_preds_test, c_test, sp_oracle)
    else
        final_results.RF_spoloss_test = missing
    end

    final_results.Absolute_spoloss_test = spo_loss(best_B_Absolute, X_test, c_test, sp_oracle)
    final_results.Huber_spoloss_test = spo_loss(best_B_Huber, X_test, c_test, sp_oracle)

    c_bar_test_preds = zeros(d_feasibleregion, n_test)
    for i = 1:n_test
        c_bar_test_preds[:, i] = c_bar_train
    end
    final_results.Baseline_spoloss_test = spo_loss(Matrix(1.0I, d_feasibleregion, d_feasibleregion), c_bar_test_preds, c_test, sp_oracle)


    # Now Hamming
    final_results.SPOplus_hammingloss_test = spo_loss(best_B_SPOplus, X_test, c_ham_test, sp_oracle)
    final_results.LS_hammingloss_test = spo_loss(best_B_leastSquares, X_test, c_ham_test, sp_oracle)
    final_results.SSVM_hammingloss_test = spo_loss(best_B_SSVM, X_test, c_ham_test, sp_oracle)

    if include_rf
        final_results.RF_hammingloss_test = spo_loss(Matrix(1.0I, d_feasibleregion, d_feasibleregion), rf_preds_test, c_ham_test, sp_oracle)
    else
        final_results.RF_hammingloss_test = missing
    end

    final_results.Absolute_hammingloss_test = spo_loss(best_B_Absolute, X_test, c_ham_test, sp_oracle)
    final_results.Huber_hammingloss_test = spo_loss(best_B_Huber, X_test, c_ham_test, sp_oracle)
    final_results.Baseline_hammingloss_test = spo_loss(Matrix(1.0I, d_feasibleregion, d_feasibleregion), c_bar_test_preds, c_ham_test, sp_oracle)

    final_results.zstar_avg_test = mean(z_test)

    return final_results
end

"""
Run multiple replications of the shortest path experiment and return a data frame.
`rng_seed` and `num_trials` are integers
`n_train_vec`, `polykernel_degree_vec`, and `polykernel_noise_half_width_vec` are
vectors so that we try all combinations of these values.
"""
function shortest_path_multiple_replications(rng_seed, num_trials, grid_dim,
    n_train_vec, n_test,
    p_features, polykernel_degree_vec, polykernel_noise_half_width_vec;
    num_lambda = 10, lambda_max = missing, lambda_min_ratio = 0.0001, holdout_percent = 0.25, regularization = :ridge,
    different_validation_losses = false,
    include_rf = true)

    Random.seed!(rng_seed)

    big_results_df = make_blank_complete_df()

    envOracle = setup_gurobi_env(method_type = :default, use_time_limit = false)
    envReform = setup_gurobi_env(method_type = :method3, use_time_limit = false)

    for n_train in n_train_vec
        for polykernel_degree in polykernel_degree_vec
            for polykernel_noise_half_width in polykernel_noise_half_width_vec
                println("Moving on to n_train = $n_train, polykernel_degree = $polykernel_degree, polykernel_noise_half_width = $polykernel_noise_half_width")
                for trial = 1:num_trials
                    println("Current trial is $trial")
                    n_holdout = round(Int, holdout_percent*n_train)

                    current_results = shortest_path_replication(grid_dim,
                        n_train, n_holdout, n_test,
                        p_features, polykernel_degree, polykernel_noise_half_width;
                        num_lambda = num_lambda, lambda_max = lambda_max, lambda_min_ratio = lambda_min_ratio, regularization = regularization,
                        gurobiEnvOracle = envOracle, gurobiEnvReform = envReform,
                        different_validation_losses = different_validation_losses,
                        include_rf = include_rf)

                    current_df_row = build_complete_row(grid_dim,
                        n_train, n_holdout, n_test,
                        p_features, polykernel_degree, polykernel_noise_half_width, current_results)

                    big_results_df = vcat(big_results_df, current_df_row)
                end
            end
        end
    end

    return big_results_df
end





### Portfolio stuff ###

"""
Run one instance of the portfolio experiment. Uses the SGD approach.
`num_lambda` is the number of lambdas used on the grid for each method with regularization.
Returns a replication_results struct.
"""
function portfolio_replication(num_assets, num_factors, n_train, n_holdout, n_test, p_features,
    n_sigmoid_polydegree, noise_multiplier_tau;
    num_lambda = 10, lambda_max = :practical, lambda_min_ratio = 0.0001,
    gurobiEnvOracle = Gurobi.Env(), gurobiEnvReform = Gurobi.Env(), different_validation_losses = false,
    data_type = :sigmoid)

    # Get oracle
    factor_size = 0.0025*noise_multiplier_tau
    sigma_noise = 0.01*noise_multiplier_tau

    L_mat = 2*factor_size*rand(num_assets, num_factors) - factor_size*ones(num_assets, num_factors)
    Sigma = L_mat*L_mat' + (sigma_noise^2)*Matrix(1.0I, num_assets, num_assets)

    avg_vec = ones(num_assets)/num_assets
    gamma = avg_vec'*Sigma*avg_vec
    gamma = 2.25*gamma

    port_oracle = portfolio_simplex_jump_setup(Sigma, gamma; gurobiEnv = gurobiEnvOracle)

    # Generate Data
    return_intercept = 0
    if data_type == :rbf_kernel
        sigma_kernel = sqrt(p_features)
        # Rademacher instead of Bernoulli
        B_true = 2*rand(Bernoulli(0.5), num_assets, p_features) - ones(num_assets, p_features)

        (X_train, c_train) = generate_rbf_kernel_returns_data(B_true, n_train, sigma_kernel, L_mat, sigma_noise, return_intercept)
        (X_validation, c_validation) = generate_rbf_kernel_returns_data(B_true, n_holdout, sigma_kernel, L_mat, sigma_noise, return_intercept)
        (X_test, c_test) = generate_rbf_kernel_returns_data(B_true, n_test, sigma_kernel, L_mat, sigma_noise, return_intercept)
    elseif data_type == :sigmoid
        bernoulli_prob = 0.5

        (X_all, c_all) = generate_sigmoid_returns_data(p_features, num_assets, n_sigmoid_polydegree, n_train + n_holdout + n_test, bernoulli_prob,
        	L_mat, sigma_noise, return_intercept)
        c_all = c_all ./ n_sigmoid_polydegree

        X_train = X_all[:, 1:n_train]
        c_train = c_all[:, 1:n_train]
        X_validation = X_all[:, (n_train + 1):(n_train + n_holdout)]
        c_validation = c_all[:, (n_train + 1):(n_train + n_holdout)]
        X_test = X_all[:, (n_train + n_holdout + 1):(n_train + n_holdout + n_test)]
        c_test = c_all[:, (n_train + n_holdout + 1):(n_train + n_holdout + n_test)]
    elseif data_type == :poly_kernel
        #### zzz need to do properly tomorrow
        B_true = rand(Bernoulli(0.5), num_assets, p_features)
        #B_true = rand(num_assets, p_features)
        polykernel_degree = n_sigmoid_polydegree

        (X_train, c_train) = generate_poly_kernel_returns_data(B_true, n_train, polykernel_degree,
        	L_mat, sigma_noise, return_intercept)
        (X_validation, c_validation) = generate_poly_kernel_returns_data(B_true, n_holdout, polykernel_degree,
        	L_mat, sigma_noise, return_intercept)
        (X_test, c_test) = generate_poly_kernel_returns_data(B_true, n_test, polykernel_degree,
        	L_mat, sigma_noise, return_intercept)
    else
        error("Enter a valid data generation type")
    end

    #println("Train returns are:")
    #println(-c_train[:,1])

    # Add intercept in the first row of X
    X_train = vcat(ones(1,n_train), X_train)
    X_validation = vcat(ones(1,n_holdout), X_validation)
    X_test = vcat(ones(1,n_test), X_test)

    # Put train + validation together
    X_both = hcat(X_train, X_validation)
    c_both = hcat(c_train, c_validation)
    train_ind = collect(1:n_train)
    validation_ind = collect((n_train + 1):(n_train + n_holdout))

    # Set validation losses
    if different_validation_losses
        spo_plus_val_loss = :spo_loss
        ls_val_loss = :least_squares_loss
        ssvm_val_loss = :hamming_loss
        absolute_val_loss = :absolute_loss
        huber_val_loss = :huber_loss
    else
        spo_plus_val_loss = :spo_loss
        ls_val_loss = :spo_loss
        ssvm_val_loss = :spo_loss
        absolute_val_loss = :spo_loss
        huber_val_loss = :spo_loss
    end

    ### Algorithms ###

    # SPO+
    best_B_SPOplus, best_lambda_SPOplus = validation_set_alg(X_both, c_both, port_oracle;
        train_ind = train_ind, validation_ind = validation_ind,
        val_alg_parms = val_parms(algorithm_type = :spo_plus_sgd, resolve_sgd = false, validation_loss = spo_plus_val_loss),
        path_alg_parms = sgd_path_parms(num_lambda = num_lambda, lambda_max = lambda_max, lambda_min_ratio = lambda_min_ratio,
            obj_accuracy = 0.001, step_type = :long_dynamic, verbose = true))

    # Least squares
    best_B_leastSquares, best_lambda_leastSquares = validation_set_alg(X_both, c_both, port_oracle;
        train_ind = train_ind, validation_ind = validation_ind,
        val_alg_parms = val_parms(algorithm_type = :ls_jump, validation_loss = ls_val_loss),
        path_alg_parms = reformulation_path_parms(num_lambda = num_lambda, lambda_max = lambda_max, regularization = :ridge,
            gurobiEnv = gurobiEnvReform, lambda_min_ratio = lambda_min_ratio, algorithm_type = :leastSquares))

    # RF
    rf_mods = train_random_forests_po(X_train, c_train;
        rf_alg_parms = rf_parms())

    # Absolute
    best_B_Absolute, best_lambda_Absolute = validation_set_alg(X_both, c_both, port_oracle;
        train_ind = train_ind, validation_ind = validation_ind,
        val_alg_parms = val_parms(algorithm_type = :ls_jump, validation_loss = absolute_val_loss),
        path_alg_parms = reformulation_path_parms(num_lambda = num_lambda, lambda_max = lambda_max, regularization = :ridge,
            po_loss_function = :absolute, gurobiEnv = gurobiEnvReform, lambda_min_ratio = lambda_min_ratio, algorithm_type = :Absolute))

    # Huber
    best_B_fake, best_lambda_fake = validation_set_alg(X_both, c_both, port_oracle;
        train_ind = train_ind, validation_ind = validation_ind,
        val_alg_parms = val_parms(algorithm_type = :ls_jump, validation_loss = ls_val_loss),
        path_alg_parms = reformulation_path_parms(lambda_max = 0.0001, regularization = :ridge,
            num_lambda = 2, gurobiEnv = gurobiEnvReform, algorithm_type = :Huber_LS_fake))

    fake_ls_list = abs.(vec(c_train - best_B_fake*X_train))
    delta_from_fake_ls = median(fake_ls_list)

    best_B_Huber, best_lambda_Huber = validation_set_alg(X_both, c_both, port_oracle;
        train_ind = train_ind, validation_ind = validation_ind,
        val_alg_parms = val_parms(algorithm_type = :ls_jump, validation_loss = huber_val_loss),
        path_alg_parms = reformulation_path_parms(num_lambda = num_lambda, lambda_max = lambda_max, regularization = :ridge, po_loss_function = :huber,
            huber_delta = delta_from_fake_ls, lambda_min_ratio = lambda_min_ratio, gurobiEnv = gurobiEnvReform, algorithm_type = :Huber))

    # Baseline
    c_bar_train = mean(c_train, dims=2)


    ### Populate final results ###
    final_results = replication_results()

    final_results.SPOplus_spoloss_test = spo_loss(best_B_SPOplus, X_test, c_test, port_oracle)
    final_results.LS_spoloss_test = spo_loss(best_B_leastSquares, X_test, c_test, port_oracle)

    rf_preds_test = predict_random_forests_po(rf_mods, X_test)
    final_results.RF_spoloss_test = spo_loss(Matrix(1.0I, num_assets, num_assets), rf_preds_test, c_test, port_oracle)

    final_results.Absolute_spoloss_test = spo_loss(best_B_Absolute, X_test, c_test, port_oracle)
    final_results.Huber_spoloss_test = spo_loss(best_B_Huber, X_test, c_test, port_oracle)

    c_bar_test_preds = zeros(num_assets, n_test)
    for i = 1:n_test
        c_bar_test_preds[:, i] = c_bar_train
    end
    final_results.Baseline_spoloss_test = spo_loss(Matrix(1.0I, num_assets, num_assets), c_bar_test_preds, c_test, port_oracle)

    (z_test, w_test) = oracle_dataset(c_test, port_oracle)
    final_results.zstar_avg_test = mean(z_test)

    return final_results
end

function portfolio_multiple_replications(rng_seed, num_trials,
    num_assets, num_factors, n_train_vec, n_test, p_features,
        n_sigmoid_polydegree_vec, noise_multiplier_tau_vec;
        num_lambda = 10, lambda_max = :practical, lambda_min_ratio = 0.0001, holdout_percent = 0.25,
        different_validation_losses = false, data_type = :poly_kernel)

    Random.seed!(rng_seed)

    big_results_df = make_blank_complete_df()

    envOracle = setup_gurobi_env(method_type = :default, use_time_limit = false)
    envReform = setup_gurobi_env(method_type = :method3, use_time_limit = false)

    for n_train in n_train_vec
        for n_sigmoid_polydegree in n_sigmoid_polydegree_vec
            for noise_multiplier_tau in noise_multiplier_tau_vec
                println("Moving on to n_train = $n_train, n_sigmoid_polydegree = $n_sigmoid_polydegree, noise_multiplier_tau = $noise_multiplier_tau")
                for trial = 1:num_trials
                    println("Current trial is $trial")
                    n_holdout = round(Int, holdout_percent*n_train)

                    current_results = portfolio_replication(num_assets, num_factors, n_train, n_holdout, n_test, p_features,
                        n_sigmoid_polydegree, noise_multiplier_tau;
                        num_lambda = num_lambda, lambda_max = lambda_max, lambda_min_ratio = lambda_min_ratio,
                        gurobiEnvOracle = envOracle, gurobiEnvReform = envReform, different_validation_losses = different_validation_losses,
                        data_type = data_type)

                    current_df_row = build_complete_row(num_assets,
                        n_train, n_holdout, n_test,
                        p_features, n_sigmoid_polydegree, noise_multiplier_tau, current_results)

                    big_results_df = vcat(big_results_df, current_df_row)
                end
            end
        end
    end

    return big_results_df
end
