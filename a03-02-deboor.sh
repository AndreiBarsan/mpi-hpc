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
#serial-deboor, parallel-deboor-a, parallel-deboor-b
mpirun -np $N_NODES -machinefile config/local-machine.txt       \
    --verbose --display-map --tag-output --timestamp-output     \
    cmake-build-debug/spline_2d_problem                         \
    --out_dir=$(pwd)/results/spline_2d_output --method parallel-deboor-b \
    --problem_sizes=30,62,126,254,510,1022 \
    --repeat 10


# TODO(andreib): Try the -xterm option on Linux!


