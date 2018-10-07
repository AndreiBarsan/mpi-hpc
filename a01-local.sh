#!/usr/bin/env bash

set -eu -o pipefail

source build_cmake.sh.inc

for N_NODES in 2 4 8 16; do
    echo -e "\nExperiment with ${N_NODES} nodes!\n\n"

    # Sum only
    mpirun -np "$N_NODES" -machinefile config/local-machine.txt $CMAKE_DIR/globsum
    # Multiple packed operations
    mpirun -np "$N_NODES" -machinefile config/local-machine.txt $CMAKE_DIR/globsum --multiple_ops
done
