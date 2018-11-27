#!/usr/bin/env bash
# Used to run the parallel DeBoor decomposition experiments from Assignment 3 locally.

################################################################################
# Preamble
################################################################################
set -eu -o pipefail

source "config.sh"
source "build_cmake.sh.inc"

N_NODES=8


################################################################################
# Main experiments
################################################################################
# Run this locally since the problem does not say we must use CDF.
mpirun -np $N_NODES -machinefile config/local-machine.txt   \
    cmake-build-debug/spline_2d_problem                     \
    --out_dir=$(pwd)/results/spline_2d_output

