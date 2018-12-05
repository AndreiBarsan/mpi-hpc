#!/usr/bin/env bash
# Used to run experiments from Assignment 2 locally (CDF not necessary).

################################################################################
# Preamble
################################################################################
set -eu -o pipefail

source "config.sh"
source "build_cmake.sh.inc"

N_NODES=2


################################################################################
# Main experiments
################################################################################
# Run this locally since the problem does not say we must use CDF.
mpirun -np $N_NODES -machinefile config/local-machine.txt   \
    --verbose --display-map --tag-output --timestamp-output \
    cmake-build-debug/spline_problem                        \
    --out_dir=$(pwd)/results/spline_output


