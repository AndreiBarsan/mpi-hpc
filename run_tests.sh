#!/usr/bin/env bash
# Used to run the parallel DeBoor decomposition experiments from Assignment 3 locally.

################################################################################
# Preamble
################################################################################
set -eu -o pipefail

source "config.sh"
source "build_cmake.sh.inc"

N_NODES=4


################################################################################
# Main tests to run with MPI
################################################################################
TESTS=(cmake-build-debug/mpi_eigen_helpers_tests)
for TEST in ${TESTS[@]}; do
    mpirun -np $N_NODES -machinefile config/local-machine.txt "$TEST"
done
