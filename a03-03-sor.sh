#!/usr/bin/env bash
# Used to run the serial SOR experiments from Assignment 3 locally.

################################################################################
# Preamble
################################################################################
set -eu -o pipefail

source "config.sh"
source "build_cmake.sh.inc"


################################################################################
# Main experiments
################################################################################

METHODS=(sor-natural)

for METHOD in ${METHODS[@]}; do
        mpirun -np 1 -machinefile config/local-machine.txt      \
            --timestamp-output         \
            cmake-build-debug/spline_2d_problem                             \
            --problem_sizes=128                              \
            --out_dir=$(pwd)/results/spline_2d_output --method "$METHOD"    \
            --repeat 1 --dump_result=true

#            --problem_sizes=8,16,32,64,128,256                              \
#           --problem_sizes=254                                              \
done

# TODO(andreib): Try the -xterm option on Linux!


