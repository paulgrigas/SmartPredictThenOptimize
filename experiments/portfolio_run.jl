using CSV, DataFrames, DecisionTree, Distributions, Gurobi, JuMP, LightGraphs, Parameters, SparseArrays, Statistics, ArgParse
include("../oracles/portfolio_oracle.jl")
include("../solver/util.jl")
include("../solver/sgd.jl")
include("../solver/reformulation.jl")
include("../solver/random_forests_po.jl")
include("../solver/validation_set.jl")
include("../experiments/replication_functions.jl")

# Fixed parameter settings (these never change)
p_features = 5
num_assets = 50
num_trials = 50
n_test = 10000
num_factors = 4

num_lambda = 1
lambda_max = 10.0^(-6)
lambda_min_ratio = 1
holdout_percent = 0.25
different_validation_losses = false
data_type = :poly_kernel


# Fixed parameter sets (these are also the same for all experiments)
n_train_vec = [100; 1000]
n_sigmoid_polydegree_vec = [1; 4; 8; 16]
noise_multiplier_tau_vec = [1; 2]

# Set this based on expt_number (40 total)
rng_seed = 2223

# Run experiment and get results
# Note that Gurobi enviornments are set within this function call
expt_results = portfolio_multiple_replications(rng_seed, num_trials,
    num_assets, num_factors, n_train_vec, n_test, p_features,
    n_sigmoid_polydegree_vec, noise_multiplier_tau_vec;
    num_lambda = num_lambda, lambda_max = lambda_max, lambda_min_ratio = lambda_min_ratio, holdout_percent = holdout_percent,
    different_validation_losses = different_validation_losses, data_type = data_type)

csv_string = "portfolio_results.csv"
CSV.write(csv_string, expt_results)
