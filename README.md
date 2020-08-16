# SmartPredictThenOptimize
Code for the paper ["Smart 'Predict, then Optimize'"](https://arxiv.org/abs/1710.08005) by Adam Elmachtoub and Paul Grigas.

The code will run on Julia v1.5.0 and requires several packages as listed in the headers of various files.

## Overview

The code is divided into several folders:
- `solver` contains all of the files needed to run the SPO+ (SGD and reformulation approaches), random forests, least squares, and least absolute loss methods.
- `oracles` contains examples of optimization oracles such as shortest path and portfolio optimization.
- `tests` contains some basic sanity check tests.
- `experiments` contains files concerning the experiments in the paper.
- `plots` contains R code for constructing the plots in the paper.

The `solver` folder contains the files `reformulation.jl` and `sgd.jl` for training a model with the SPO+ loss function using a fixed value of the regularization parameter. To actually a train a model in practice, it is recommended to use the file `validation_set.jl` and the corresponding function is `validation_set_alg`, which additionally performs cross-validation using the validation set approach. The documentation in the file explains the syntax for calling the `validation_set_alg` function.

The files in the folder `experiments` can simply be run to replicate the experiments included in the paper.
