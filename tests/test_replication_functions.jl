include("../oracles/shortest_path_oracle.jl")
include("../oracles/portfolio_oracle.jl")
include("../solver/util.jl")
include("../solver/sgd.jl")
include("../solver/reformulation.jl")
include("../solver/random_forests_po.jl")
include("../solver/validation_set.jl")
include("../experiments/replication_functions.jl")

using Random, Distributions, Test, CSV

function test_sp_replication()
    Random.seed!(8392)

    envOracle = setup_gurobi_env(method_type = :default, use_time_limit = true)
    envReform = setup_gurobi_env(method_type = :method3, use_time_limit = true)

    final_results = shortest_path_replication(5,
        100, 25, 10000,
        5, 8, 0.5;
        num_lambda = 10, lambda_max = 1.0, lambda_min_ratio = 10.0^(-8), regularization = :lasso,
        gurobiEnvOracle = envOracle, gurobiEnvReform = envReform,
        different_validation_losses = false)

    println(final_results)
end

function test_port_replication()
    Random.seed!(780)

    envOracle = setup_gurobi_env(method_type = :default, use_time_limit = true)
    envReform = setup_gurobi_env(method_type = :method3, use_time_limit = true)

    final_results = portfolio_replication(200, 4, 1000, 250, 10000, 5,
        8, 1;
        num_lambda = 10, lambda_max = 100.0, lambda_min_ratio = 10.0^(-8),
        gurobiEnvOracle = envOracle, gurobiEnvReform = envReform, different_validation_losses = false,
        data_type = :poly_kernel)

    println(final_results)
end
