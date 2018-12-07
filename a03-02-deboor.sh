#!/usr/bin/env bash
# Used to run the parallel DeBoor decomposition experiments from Assignment 3 locally.

################################################################################
# Preamble
################################################################################
set -eu -o pipefail

source "config.sh"
source "build_cmake.sh.inc"


################################################################################
# Main experiments
################################################################################
# Run this locally since the problem does not say we must use CDF.

#METHODS=(parallel-deboor-a parallel-deboor-b)
NODE_COUNTS=(1 2 4 8 16)
METHODS=(parallel-deboor-a)
#NODE_COUNTS=(4)

for METHOD in ${METHODS[@]}; do
    for NODE_COUNT in ${NODE_COUNTS[@]}; do
        echo -e "\n\tRunning with $METHOD on $NODE_COUNT nodes...\n\n\n\n"
        mpirun -np "$NODE_COUNT" -machinefile config/local-machine.txt      \
            --verbose --display-map --tag-output --timestamp-output         \
            cmake-build-debug/spline_2d_problem                             \
            --out_dir=$(pwd)/results/spline_2d_output --method "$METHOD"    \
            --problem_sizes=254                                             \
            --repeat 4 --dump_result=false

#            --problem_sizes=30,62,126,254,510,1022                          \
    done
done

# TODO(andreib): Try the -xterm option on Linux!


