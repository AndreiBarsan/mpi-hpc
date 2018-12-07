#!/usr/bin/env bash
# Used to run the parallel DeBoor decomposition experiments from Assignment 3 locally.

################################################################################
# Preamble
################################################################################
set -eu -o pipefail

source "config.sh"
source "build_cmake.sh.inc"


################################################################################
# Setup and Orchestration
################################################################################
echo "Will rsync code to CDF..."
time rsync -rv --exclude 'cmake-build-debug' \
    --exclude 'results' --exclude '.git' --exclude '.idea' \
     . "${RPROJ}"
echo "OK"

START_NODE="$(cat cdf_machines | tail -n 1)"
RH="${CDF_USER}@${START_NODE}"

echo "Will SSH to $RH. If you can't connect, make sure you are on the UofT network, e.g., using the VPN!"
echo "You may have to manually connect first and run 'kinit' MPI can work."

echo -e "\n\n NOT REBUILDING \n "
# ssh "$RH" "hpsc/build.sh"

# The sed tails everything except the last line.
# We use the first line as the starting node, and the rest as workers.
ssh "$RH" cat "hpsc/cdf_machines | sed '\$d' >| /tmp/machines"




################################################################################
# Main experiments
################################################################################
# Run this locally since the problem does not say we must use CDF.

#METHODS=(parallel-deboor-a parallel-deboor-b)
NODE_COUNTS=(2 4 8 16 32)
METHODS=(parallel-deboor-a)
#NODE_COUNTS=(4)

ssh "$RH" kinit

for METHOD in ${METHODS[@]}; do
    for NODE_COUNT in ${NODE_COUNTS[@]}; do
        echo -e "\n\tRunning with $METHOD on $NODE_COUNT nodes...\n\n\n\n"

        # ssh "$RH" mpirun -np $N_NODES -pernode -machinefile /tmp/machines hpsc/build/globsum --out_dir=hpsc/results
        ssh "$RH" mpirun -np "$NODE_COUNT" -pernode -machinefile /tmp/machines \
            --verbose --display-map --tag-output --timestamp-output         \
            hpsc/build/spline_2d_problem                             \
            --out_dir=$(pwd)/results/spline_2d_output --method "$METHOD"    \
            --problem_sizes=510                                             \
            --repeat 5 --dump_result=false

#            --problem_sizes=30,62,126,254,510,1022                          \
    done
done

# TODO(andreib): Try the -xterm option on Linux!


