# SmartPredictThenOptimize
Code for the paper ["Smart 'Predict, then Optimize'"](https://arxiv.org/abs/1710.08005) by Adam Elmachtoub and Paul Grigas.

## Overview

The code is divided into several folders:
- `solver` contains all of the files needed to run the method.
- `oracles` contains examples of optimization oracles.
- `tests` contains some basic sanity check tests.
- `experiments` contains files concerning the experiments in the paper.
- `plots` contains R code for constructing the plots in the paper.

The main file in the `solver` folder for training a model with the SPO+ loss function is `validation_set.jl` and the corresponding function is `validation_set_alg`. The documentation in the file explains the syntax for calling this function.

The files in the folder
