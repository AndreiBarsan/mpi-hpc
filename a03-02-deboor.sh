#!/usr/bin/env bash
# Used to run the parallel DeBoor decomposition experiments from Assignment 3 locally.

################################################################################
# Preamble
################################################################################
set -eu -o pipefail

source "config.sh"
source "build_cmake.sh.inc"

#N_NODES=8


################################################################################
# Main experiments
################################################################################
# Run this locally since the problem does not say we must use CDF.

# XXX(andreib): ALSO RUN WITH 16 procs on 32x32 system!!
METHODS=(parallel-deboor-a parallel-deboor-b)
#NODE_COUNTS=(1 2 4 8 16)
#METHODS=(parallel-deboor-b)
NODE_COUNTS=(16)

for METHOD in ${METHODS[@]}; do
    for NODE_COUNT in ${NODE_COUNTS[@]}; do
        echo -e "\n\tRunning with $METHOD on $NODE_COUNT nodes...\n\n\n\n"
        mpirun -np "$NODE_COUNT" -machinefile config/local-machine.txt       \
            --verbose --display-map --tag-output --timestamp-output     \
            cmake-build-debug/spline_2d_problem                         \
            --out_dir=$(pwd)/results/spline_2d_output --method "$METHOD" \
            --problem_sizes=30 \
            --repeat 12 --dump_result=false

#            --problem_sizes=62,126,254,510,1022,2046 \
    done
done

# TODO(andreib): Try the -xterm option on Linux!


