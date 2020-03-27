#=
Random Forests in the "Predict, then Optimize" (PO) setting. Uses package DecisionTree.
=#

using Parameters, DecisionTree

## Type for Random Forest algorithm parameters with default values
@with_kw struct rf_parms
    num_trees::Int = 100
    num_features_per_split::Int = -1
end

"""
    train_random_forests_po(X::Matrix{Float64}, c::Matrix{Float64};
        num_trees = 100, num_features_per_split = -1)

Train d different Random Forests regression models, each to predict a different
component of the cost vector as a function of the features. Returns a list of the
d different RF models each trained from the `DecisionTree` package.

# Arguments
- `X`: p x n training set feature matrix
- `c`: d x n training set matrix of cost vectors
- `num_trees` is the total number of trees in each RF model
- `num_features_per_split` is the number of randomly sampled features at each split
"""
function train_random_forests_po(X::Matrix{Float64}, c::Matrix{Float64};
    rf_alg_parms::rf_parms = rf_parms())

    @unpack num_trees, num_features_per_split = rf_alg_parms

    (p, n) = size(X)
    (d, n2) = size(c)
    if n != n2
        error("Dimensions of the input are mismatched.")
    end

    X_t = copy(transpose(X))

    # If num_features_per_split is not specified, use default for regression
    if num_features_per_split < 1
        num_features_per_split = ceil(Int, p/3)
    end

    rf_model_list = DecisionTree.Ensemble[]

    # train one model for each component
    for j = 1:d
        c_vec = c[j, :]
        rf_model = build_forest(c_vec, X_t, num_features_per_split, num_trees)
        push!(rf_model_list, rf_model)
    end

    return rf_model_list
end

"""
    predict_random_forests_po(rf_model_list, X_new)

Given a list of random forests models trained previously by the `run_random_forests_po`
function and a p x n matrix of new features values, return a d x n matrix of predictions.

# Arguments
- `rf_model_list` is the length d list of RF model objects, each to predict a different component of the cost vector based on a feature vector of length p.
- `X_new`: p x n feature matrix. Each column is one of n observations.
"""
function predict_random_forests_po(rf_model_list, X_new)
    (p, n) = size(X_new)
    d = length(rf_model_list)

    X_new_t = copy(transpose(X_new))

    preds = zeros(d, n)
    for j = 1:d
        preds[j, :] = apply_forest(rf_model_list[j], X_new_t)
    end

    return preds
end
