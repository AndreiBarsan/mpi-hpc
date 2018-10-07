#!/usr/bin/env bash

set -eu -o pipefail

source build_cmake.sh.inc

for N_NODES in 2 4 8 16; do
    mpirun -np "$N_NODES" -machinefile config/local-machine.txt $CMAKE_DIR/globsum
done
