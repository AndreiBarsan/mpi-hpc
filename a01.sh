#!/usr/bin/env bash

set -eu -o pipefail

N_NODES=4

# TODO make this into a makefile maybe
mpirun -np "$N_NODES" -machinefile config/local-machine.txt cmake-build-debug/globsum