using CSV, DataFrames, DecisionTree, Distributions, Gurobi, JuMP, LightGraphs, Parameters, SparseArrays, Statistics, ArgParse
include("../oracles/shortest_path_oracle.jl")
include("../solver/util.jl")
include("../solver/sgd.jl")
include("../solver/reformulation.jl")
include("../solver/random_forests_po.jl")
include("../solver/validation_set.jl")
include("../experiments/replication_functions.jl")

# Fixed parameter settings (these never change)
p_features = 5
grid_dim = 5
num_trials = 50
n_test = 10000

num_lambda = 1
lambda_max = 0
lambda_min_ratio = 1
holdout_percent = 0.25
regularization = :lasso
different_validation_losses = false
include_rf = true


# Fixed parameter sets (these are also the same for all experiments)
n_train_vec = [5000]
polykernel_degree_vec = [1; 2; 4; 6; 8]
polykernel_noise_half_width_vec = [0; 0.5]

# Set this to get reproducible results
rng_seed = 64234

# Run experiment and get results
# Note that Gurobi enviornments are set within this function call
expt_results = shortest_path_multiple_replications(rng_seed, num_trials, grid_dim,
    n_train_vec, n_test,
    p_features, polykernel_degree_vec, polykernel_noise_half_width_vec;
    num_lambda = num_lambda, lambda_max = lambda_max, lambda_min_ratio = lambda_min_ratio,
    holdout_percent = holdout_percent, regularization = regularization,
    different_validation_losses = different_validation_losses,
    include_rf = include_rf)

csv_string = "shortest_path_5000.csv"
CSV.write(csv_string, expt_results)
