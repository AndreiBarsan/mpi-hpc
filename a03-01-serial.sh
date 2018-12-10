#!/usr/bin/env bash
# Used to run the serial DeBoor decomposition experiments from Assignment 3.

################################################################################
# Preamble
################################################################################
set -eu -o pipefail

source "config.sh"
source "build_cmake.sh.inc"


################################################################################
# Main experiments
################################################################################

cmake-build-debug/spline_2d_problem                                     \
    --problem_sizes=8,16,32,64,128,256                                  \
    --out_dir=$(pwd)/results/spline_2d_output --method serial-deboor    \
    --repeat 12 --dump_result=false

#           --problem_sizes=254                                              \


