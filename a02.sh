#!/usr/bin/env bash
# Used to run experiments from Assignment 1 on the CDF.

################################################################################
# Preamble
################################################################################
set -eu -o pipefail

source "config.sh"
source "build_cmake.sh.inc"

N_NODES=4;


################################################################################
# Main experiments
################################################################################
# Run this locally since the problem does not say we must use CDF.
mpirun -np $N_NODES -machinefile config/local-machine.txt cmake-build-debug/spline_problem \
    --out_dir=$(pwd)/results/spline_output

#if [[ "$PROBLEM" == "2" ]]; then
#    for N_NODES in 2 4 8 16; do
#        echo -e "\n\n\tExperiment with ${N_NODES} nodes!\n\n"
#
#        ssh "$RH" mpirun -np $N_NODES -pernode -machinefile /tmp/machines hpsc/build/globsum --out_dir=hpsc/results --multiple_ops
#    done
#else
#    for N_NODES in 2 4 8 16; do
#        echo -e "\n\n\tExperiment with ${N_NODES} nodes!\n\n"
#
#        ssh "$RH" mpirun -np $N_NODES -pernode -machinefile /tmp/machines hpsc/build/ring --out_dir=hpsc/results --iterations=250
#    done
#fi
#
#echo "Grabbing results back for number crunching..."
#time rsync -rv "$RH:hpsc/results/" "results/cdf/"

