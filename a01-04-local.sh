#!/usr/bin/env bash

set -eu -o pipefail

source build_cmake.sh.inc
echo "Running A04 Problem 04 experiments LOCALLY"

cd $CMAKE_DIR

for N_NODES in 2 4 8 16; do
    echo -e "\nExperiment with ${N_NODES} nodes!\n\n\n\n"

    # Sum only
    mpirun -np "$N_NODES" -machinefile ../config/local-machine.txt ring --iterations=50
done
