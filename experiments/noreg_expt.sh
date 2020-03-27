#!/bin/bash
# Job name:
#SBATCH --job-name=spo_shortestpath_noreg_march2020
#
# Partition:
#SBATCH --partition=savio
#
# Request one node:
#SBATCH --nodes=1
#
# Specify number of tasks for use case:
#SBATCH --ntasks-per-node=20
#
# Processors per task:
#SBATCH --cpus-per-task=1
#
# Wall clock limit:
#SBATCH --time=20:00:00
#
# Mail type:
#SBATCH --mail-type=all
#
# Mail user:
#SBATCH --mail-user=pgrigas@berkeley.edu
#
## Command(s) to run:
julia shortest_path_run_noreg.jl
